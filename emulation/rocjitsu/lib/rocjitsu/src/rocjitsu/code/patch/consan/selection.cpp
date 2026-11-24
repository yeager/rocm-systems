// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/prepare.h"

#include "rocjitsu/code/consan/image/artifact.h"
#include "util/diagnostic.h"
#include "util/log.h"
#include "util/utf8.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace rocjitsu::consan {
namespace {

bool unsupported_entry(InputErrorCode code) {
  switch (code) {
  case InputErrorCode::UnsupportedEntry:
  case InputErrorCode::UnsupportedAccess:
  case InputErrorCode::UnsupportedSynchronization:
  case InputErrorCode::UnsupportedInstruction:
  case InputErrorCode::UnsupportedFlow:
    return true;
  default:
    return false;
  }
}

std::string_view reason_name(InputErrorCode code) {
  switch (code) {
  case InputErrorCode::None:
    return "none";
  case InputErrorCode::InvalidImage:
    return "invalid_image";
  case InputErrorCode::UnsupportedTarget:
    return "unsupported_target";
  case InputErrorCode::UnsupportedEntry:
    return "unsupported_entry";
  case InputErrorCode::UnsupportedAccess:
    return "unsupported_access";
  case InputErrorCode::UnsupportedSynchronization:
    return "unsupported_synchronization";
  case InputErrorCode::UnsupportedInstruction:
    return "unsupported_instruction";
  case InputErrorCode::UnsupportedFlow:
    return "unsupported_flow";
  case InputErrorCode::UnsupportedControlTransferRange:
    return "unsupported_control_transfer_range";
  case InputErrorCode::DecodeFailure:
    return "decode_failure";
  case InputErrorCode::ResourceLimit:
    return "resource_limit";
  }
  return "unknown";
}

void json_string(std::ostream &out, std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\')
      out << '\\' << ch;
    else if (ch < 32)
      out << "\\u00" << digits[ch >> 4] << digits[ch & 15];
    else
      out << ch;
  }
  out << '"';
}

} // namespace

FailureOr<PreparationResult> prepare_image_best_effort(std::span<const uint8_t> original,
                                                       const TargetProfile &target,
                                                       std::span<const std::string_view> entries,
                                                       InputError &error, ScopeKind scope) {
  error = {};
  const InputLimits limits;
  if (entries.size() > kImageEntryLimit || original.size() > limits.image_bytes) {
    error = {.code = InputErrorCode::ResourceLimit,
             .text_offset = {},
             .message = "selected input copies exceed bounded image analysis budget"};
    return Result::failure();
  }
  const TargetProfile *canonical = target_profile(target.name);
  if (canonical == nullptr || *canonical != target) {
    error = {.code = InputErrorCode::UnsupportedTarget,
             .text_offset = {},
             .message = "unsupported or inconsistent native LDS target profile"};
    return Result::failure();
  }
  if (scope != ScopeKind::NativeLdsNoSynchronization && scope != ScopeKind::ExecutionTrace) {
    error = {.code = InputErrorCode::UnsupportedEntry,
             .text_offset = {},
             .message = "unknown preparation profile"};
    return Result::failure();
  }
  std::unordered_set<std::string_view> names;
  for (const std::string_view entry : entries) {
    if (!util::is_valid_utf8(entry) || entry.find('\0') != std::string_view::npos) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "entry name is not valid UTF-8 for JSON output"};
      return Result::failure();
    }
    if (!names.insert(entry).second) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "duplicate selected entry name"};
      return Result::failure();
    }
  }
  // Native inspection establishes a complete valid original before any local
  // qualification failure can be treated as an unsupported entry.
  util::StringDiagnostic diagnostic;
  const FailureOr<NativeImageInventory> inventory =
      inspect_native_image(original, target.name, diagnostic.emitter());
  if (inventory.failed()) {
    error = {
        .code = InputErrorCode::InvalidImage, .text_offset = {}, .message = diagnostic.message()};
    return Result::failure();
  }
  std::vector<std::string_view> automatic_entries;
  if (entries.empty()) {
    automatic_entries.reserve(inventory.value().entries.size());
    for (const ImageEntryFacts &entry : inventory.value().entries)
      automatic_entries.push_back(entry.binding.name);
    entries = automatic_entries;
  }
  if (original.size() > limits.image_bytes / entries.size()) {
    error = {.code = InputErrorCode::ResourceLimit,
             .text_offset = {},
             .message = "selected input copies exceed bounded image analysis budget"};
    return Result::failure();
  }
  for (const std::string_view entry : entries) {
    if (std::ranges::find(inventory.value().entries, entry, [](const ImageEntryFacts &facts) {
          return facts.binding.name;
        }) == inventory.value().entries.end()) {
      error = {.code = InputErrorCode::UnsupportedEntry,
               .text_offset = {},
               .message = "requested kernel entry '" + std::string(entry) + "' was not found"};
      return Result::failure();
    }
  }
  PreparationResult result = {.original = inventory.value().identity,
                              .target = target.id,
                              .requested_scope = scope,
                              .entries = {},
                              .artifact = std::nullopt};
  std::vector<std::string_view> selected;
  for (const std::string_view name : entries) {
    const std::vector<ImageEntryFacts>::const_iterator binding =
        std::ranges::find(inventory.value().entries, name,
                          [](const ImageEntryFacts &facts) { return facts.binding.name; });
    EntryPreparation entry = {
        .entry = std::string(name), .key = binding->key, .instrumented = false, .reason = {}};
    InputError reason;
    const FailureOr<InputProgram> program =
        build_image_entry_program(original, target, name, reason, {}, scope);
    if (program.failed()) {
      if (!unsupported_entry(reason.code)) {
        error = std::move(reason);
        return Result::failure();
      }
      entry.reason = std::move(reason);
      util::Logger::consan("retaining native entry ", name, ": ", entry.reason.message);
    } else {
      entry.instrumented = true;
      selected.push_back(name);
      util::Logger::consan("qualified entry ", name);
    }
    result.entries.push_back(std::move(entry));
  }
  if (!selected.empty()) {
    FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, selected, error, scope);
    if (artifact.succeeded()) {
      result.artifact = std::move(artifact.value());
    } else {
      // Individual ISA qualification already succeeded. Only recovery-marked
      // callable capability failures may retry a late UnsupportedInstruction;
      // native preservation uses the same code and must remain a hard failure.
      if (error.code != InputErrorCode::UnsupportedControlTransferRange &&
          !(error.code == InputErrorCode::UnsupportedInstruction &&
            error.recovery == InputErrorRecovery::NativeCallableCapability))
        return Result::failure();
      std::vector<std::string_view> accepted;
      for (EntryPreparation &entry : result.entries) {
        if (!entry.instrumented)
          continue;
        accepted.push_back(entry.entry);
        InputError candidate_error;
        FailureOr<PreparedImageArtifact> candidate =
            prepare_image(original, target, accepted, candidate_error, scope);
        if (candidate.succeeded()) {
          result.artifact = std::move(candidate.value());
          continue;
        }
        if (candidate_error.code != InputErrorCode::UnsupportedControlTransferRange &&
            !(candidate_error.code == InputErrorCode::UnsupportedInstruction &&
              candidate_error.recovery == InputErrorRecovery::NativeCallableCapability)) {
          error = std::move(candidate_error);
          return Result::failure();
        }
        accepted.pop_back();
        entry.instrumented = false;
        entry.reason = std::move(candidate_error);
        util::Logger::consan("retaining native entry ", entry.entry, ": ", entry.reason.message);
      }
      error = {};
    }
  }
  return result;
}

std::string render_preparation_report(const PreparationResult &result) {
  const size_t instrumented = static_cast<size_t>(std::ranges::count_if(
      result.entries, [](const EntryPreparation &entry) { return entry.instrumented; }));
  const size_t skipped = result.entries.size() - instrumented;
  std::ostringstream out;
  out << "{\"schema_version\":1,\"outcome\":\""
      << (result.artifact ? "instrumented" : "native_only") << "\",\"original_sha256\":\"";
  constexpr char digits[] = "0123456789abcdef";
  for (const uint8_t byte : result.original.sha256)
    out << digits[byte >> 4] << digits[byte & 15];
  out << "\",\"selection_profile\":";
  if (result.artifact) {
    out << '"';
    for (const uint8_t byte : result.artifact->facts.selection_profile)
      out << digits[byte >> 4] << digits[byte & 15];
    out << '"';
  } else {
    out << "null";
  }
  out << ",\"original_bytes\":" << result.original.bytes << ",\"target\":" << result.target
      << ",\"requested_scope\":\""
      << (result.requested_scope == ScopeKind::ExecutionTrace ? "execution_trace" : "lds")
      << "\",\"requested\":" << result.entries.size() << ",\"instrumented\":" << instrumented
      << ",\"skipped\":" << skipped << ",\"requested_coverage\":\""
      << (instrumented == 0 ? "not_performed" : (skipped == 0 ? "complete" : "partial"))
      << "\",\"entries\":[";
  bool first = true;
  for (const EntryPreparation &entry : result.entries) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"entry\":";
    json_string(out, entry.entry);
    out << ",\"entry_key\":" << entry.key << ",\"disposition\":\""
        << (entry.instrumented ? "instrumented" : "native_uncovered") << "\",\"reason_code\":";
    json_string(out, reason_name(entry.reason.code));
    out << ",\"text_offset\":";
    if (entry.reason.text_offset)
      out << *entry.reason.text_offset;
    else
      out << "null";
    out << ",\"reason\":";
    json_string(out, entry.reason.message);
    out << '}';
  }
  out << "]}\n";
  return out.str();
}

} // namespace rocjitsu::consan
