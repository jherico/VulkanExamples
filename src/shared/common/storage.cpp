//
//  Created by Bradley Austin Davis on 2016/02/17
//  Copyright 2013-2017 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#include "storage.hpp"
#include <cstring>
#include <fstream>
#include <istream>
#include <iterator>
#include <string>

#if defined(WIN32)
#include <Windows.h>
#endif

namespace vks { namespace storage {

#if defined(__ANDROID__)
AAssetManager* assetManager = nullptr;
void setAssetManager(AAssetManager* assetManager) {
    vks::storage::assetManager = assetManager;
}
#endif

class MemoryStorage : public Storage {
public:
    MemoryStorage(Span& span)
        : MemoryStorage(span.size()) {
        memcpy(_data.data(), span.data(), span.size());
    }

    MemoryStorage(size_t size) { _data.resize(size); }

    Span span() const override { return Span(_data); }

private:
    std::vector<uint8_t> _data;
};

#if defined(__ANDROID__) || defined(WIN32)
#define MAPPED_FILES 1
#else
#define MAPPED_FILES 0
#endif

#if MAPPED_FILES

class FileStorage : public Storage {
public:
    static StoragePointer create(const std::string& filename, Span& span);
    FileStorage(const std::string& filename);
    ~FileStorage();
    // Prevent copying
    FileStorage(const FileStorage& other) = delete;
    FileStorage& operator=(const FileStorage& other) = delete;

    Span span() const override { return Span(_mapped, _size); }

private:
    size_t _size{ 0 };
    uint8_t* _mapped{ nullptr };
#if defined(__ANDROID__)
    AAsset* _asset{ nullptr };
#elif (WIN32)
    HANDLE _file{ INVALID_HANDLE_VALUE };
    HANDLE _mapFile{ INVALID_HANDLE_VALUE };
#else
    std::vector<uint8_t> _data;
#endif
};

FileStorage::FileStorage(const std::string& filename) {
#if defined(__ANDROID__)
    // Load shader from compressed asset
    _asset = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_BUFFER);
    assert(_asset);
    _size = AAsset_getLength(_asset);
    assert(_size > 0);
    _mapped = (uint8_t*)(AAsset_getBuffer(_asset));
#elif (WIN32)
    _file = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (_file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file");
    }
    {
        DWORD dwFileSizeHigh;
        _size = GetFileSize(_file, &dwFileSizeHigh);
        _size += (((size_t)dwFileSizeHigh) << 32);
    }
    _mapFile = CreateFileMappingA(_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (_mapFile == INVALID_HANDLE_VALUE || _mapFile == nullptr) {
        throw std::runtime_error("Failed to create mapping");
    }
    _mapped = (uint8_t*)MapViewOfFile(_mapFile, FILE_MAP_READ, 0, 0, 0);
#endif
}

FileStorage::~FileStorage() {
#if defined(__ANDROID__)
    AAsset_close(_asset);
#elif (WIN32)
    UnmapViewOfFile(_mapped);
    CloseHandle(_mapFile);
    CloseHandle(_file);
#endif
}

#endif

StoragePointer Storage::create(Span& span) {
    return std::make_shared<MemoryStorage>(span);
}
StoragePointer Storage::readFile(const std::string& filename) {
#if MAPPED_FILES
    return std::make_shared<FileStorage>(filename);
#else
    // FIXME move to posix memory mapped files
    // open the file:
    std::ifstream file(filename, std::ios::binary);
    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> fileData;
    // reserve capacity
    fileData.reserve(fileSize);
    // read the data:
    fileData.insert(fileData.begin(), std::istream_iterator<uint8_t>(file), std::istream_iterator<uint8_t>());
    file.close();
    return std::make_shared<MemoryStorage>(fileData.size(), fileData.data());
#endif
}

}}  // namespace vks::storage
