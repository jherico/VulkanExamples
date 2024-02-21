//
//  Created by Bradley Austin Davis on 2016/02/17
//  Copyright 2013-2017 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#pragma once

#include <stdint.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <span>

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#endif

namespace vks { namespace storage {

#if defined(__ANDROID__)
void setAssetManager(AAssetManager* assetManager);
#endif

class Storage;
using StoragePointer = std::shared_ptr<const Storage>;
using ByteArray = std::vector<uint8_t>;
using Span = std::span<const uint8_t>;

// Abstract class to represent memory that stored _somewhere_ (in system memory or in a file, for example)
class Storage : public std::enable_shared_from_this<Storage> {
public:
    virtual ~Storage() {}
    virtual Span span() const = 0;

    static StoragePointer create(Span& span);
    static StoragePointer readFile(const std::string& filename);
};

}}  // namespace vks::storage
