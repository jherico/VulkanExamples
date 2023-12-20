#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vks { namespace file {

using SimpleHandler = std::function<void(size_t, const void*)>;
using NamedHandler = std::function<void(const char*, size_t, const void*)>;

void withBinaryFileContents(const std::string& filename, const SimpleHandler& handler);

void withBinaryFileContents(const std::string& filename, const NamedHandler& handler);

std::string readTextFile(const std::string& fileName);

}}  // namespace vks::file
