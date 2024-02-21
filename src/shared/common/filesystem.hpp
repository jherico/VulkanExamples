#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <span>

namespace vks { namespace file {

using Span = std::span<const uint8_t>;
using SimpleHandler = std::function<void(Span)>;
using NamedHandler = std::function<void(const char*, Span)>;

void withBinaryFileContents(const std::string& filename, const SimpleHandler& handler);

void withBinaryFileContents(const std::string& filename, const NamedHandler& handler);

std::string readTextFile(const std::string& fileName);

}}  // namespace vks::file
