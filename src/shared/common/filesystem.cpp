#include "filesystem.hpp"

#include <cstring>
#include <fstream>
#include <istream>
#include <iterator>
#include <functional>

#include "storage.hpp"

namespace vks { namespace file {

void withBinaryFileContents(const std::string& filename, const SimpleHandler& handler) {
    withBinaryFileContents(filename, [&handler](const char*, size_t size, const void* data_) { handler(size, data_); });
}

void withBinaryFileContents(const std::string& filename, const NamedHandler& handler) {
    auto storage = storage::Storage::readFile(filename);
    handler(filename.c_str(), storage->size(), storage->data());
}

std::vector<uint8_t> readBinaryFile(const std::string& filename) {
    std::vector<uint8_t> result;
    withBinaryFileContents(filename, [&result](size_t size, const void* data) {
        result.resize(size);
        memcpy(result.data(), data, size);
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
