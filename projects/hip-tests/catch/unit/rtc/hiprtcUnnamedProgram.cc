/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <hip/hiprtc.h>

#include <sys/inotify.h>
#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <string>

/**
 * @addtogroup hiprtcCreateProgram hiprtcCreateProgram
 * @{
 * @ingroup hiprtc
 */

/**
 * Test Description
 * ------------------------
 *  - hiprtcCreateProgram without a program name generates a unique name in memory: it must not
 *    create (or create and remove) a file in the current directory, which may be read-only or
 *    shared by many processes. The current directory is an empty temporary directory watched
 *    with inotify.
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtcUnnamedProgram.cc
 * Test requirements
 * ------------------------
 *  - Linux
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hiprtcCreateProgram_UnnamedLeavesCwdUntouched) {
  char oldCwd[PATH_MAX];
  REQUIRE(getcwd(oldCwd, sizeof(oldCwd)) != nullptr);
  char dir[] = "/tmp/hiprtc_unnamed_XXXXXX";
  REQUIRE(mkdtemp(dir) != nullptr);

  // RAII guard: restore the original working directory and remove the temp dir
  // even when a REQUIRE or HIPRTC_CHECK assertion unwinds the stack.
  struct Cleanup {
    const char* oldCwd;
    const char* dir;
    int fd = -1;
    ~Cleanup() {
      if (fd >= 0) close(fd);
      (void)chdir(oldCwd);
      rmdir(dir);
    }
  } cleanup{oldCwd, dir};

  REQUIRE(chdir(dir) == 0);

  cleanup.fd = inotify_init1(IN_NONBLOCK);
  REQUIRE(cleanup.fd >= 0);
  REQUIRE(inotify_add_watch(cleanup.fd, dir, IN_CREATE) >= 0);

  static constexpr const char* kSource = R"(extern "C" __global__ void k(int* a) { a[0] = 1; })";
  hiprtcProgram programs[2] = {};
  // Two unnamed programs: each gets a generated name.
  for (auto& program : programs) {
    HIPRTC_CHECK(hiprtcCreateProgram(&program, kSource, nullptr, 0, nullptr, nullptr));
  }
  HIPRTC_CHECK(hiprtcCompileProgram(programs[0], 0, nullptr));
  for (auto& program : programs) {
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  }

  alignas(inotify_event) char events[4096];
  const ssize_t bytes = read(cleanup.fd, events, sizeof(events));
  // fd and dir are released by ~Cleanup; restore cwd before the final REQUIRE.
  REQUIRE(chdir(oldCwd) == 0);

  INFO("inotify read " << bytes << " bytes of IN_CREATE events in the current directory");
  REQUIRE(bytes <= 0);
}

/**
 * End doxygen group hiprtcCreateProgram.
 * @}
 */
