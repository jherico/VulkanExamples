#include "filesystem.hpp"

#include <cstring>
#include <fstream>
#include <functional>
#include <istream>
#include <iterator>
#include <span>

#include "storage.hpp"

namespace vks { namespace file {

void withBinaryFileContents(const std::string& filename, const SimpleHandler& handler) {
    NamedHandler namedHandler = [&handler](const char*, Span span) { handler(span); };
    withBinaryFileContents(filename, namedHandler);
}

void withBinaryFileContents(const std::string& filename, const NamedHandler& handler) {
    auto storage = storage::Storage::readFile(filename);
    vks::file::Span span = storage->span();
    handler(filename.c_str(), span);
}

std::vector<uint8_t> readBinaryFile(const std::string& filename) {
    std::vector<uint8_t> result;
    withBinaryFileContents(filename, [&result](Span span) {
        result.resize(span.size());
        memcpy(result.data(), span.data(), span.size());
    });
    return result;
}

std::string readTextFile(const std::string& fileName) {
    std::string fileContent;
    std::ifstream fileStream(fileName, std::ios::in);

    if (!fileStream.is_open()) {
        throw std::invalid_argument("File " + fileName + " not found");
    }
    std::string line = "";
    while (!fileStream.eof()) {
        getline(fileStream, line);
        fileContent.append(line + "\n");
    }
    fileStream.close();
    return fileContent;
}

}}  // namespace vks::file
