#include "device.hpp"

vk::DeviceSize vks::DeviceProperties::getAlignedSize(vk::DeviceSize alignment, vk::DeviceSize size) {
    if (0 == alignment || 0 == (size % alignment)) {
        return size;
    }
    return ((size + alignment - 1) / alignment) * alignment;
}

vk::DeviceSize vks::DeviceProperties::getUniformAlignedSize(vk::DeviceSize size) const {
    return getAlignedSize(core10.limits.minUniformBufferOffsetAlignment, size);
}

vk::DeviceSize vks::DeviceProperties::getUniformAlignedOffset(vk::DeviceSize size, size_t count) const {
    return getUniformAlignedSize(size) * count;
}

vk::DeviceSize vks::DeviceProperties::getTexelAlignedSize(vk::DeviceSize size) const {
    return getAlignedSize(core10.limits.minTexelBufferOffsetAlignment, size);
}

vk::DeviceSize vks::DeviceProperties::getTexelAlignedOffset(vk::DeviceSize size, size_t count) const {
    return getTexelAlignedSize(size) * count;
}

vk::DeviceSize vks::DeviceProperties::getStorageAlignedSize(vk::DeviceSize size) const {
    return getAlignedSize(core10.limits.minStorageBufferOffsetAlignment, size);
}

vk::DeviceSize vks::DeviceProperties::getStorageAlignedOffset(vk::DeviceSize size, size_t count) const {
    return getStorageAlignedSize(size) * count;
}

vks::DeviceInfo::DeviceInfo(const vk::PhysicalDevice& physcialDevice) {
    static ::vk::PhysicalDevice EMPTY = {};
    if (physcialDevice) {
        extensions = getExtensions(physcialDevice);
        features.load(physcialDevice, extensions);
        properties.load(physcialDevice, extensions);
        memoryProperties.load(physcialDevice, extensions);
        supportedDepthFormat = getSupportedDepthFormat(physcialDevice);
    }
}

vk::Format vks::DeviceInfo::getSupportedDepthFormat(const vk::PhysicalDevice& physicalDevice) {
    // Since all depth formats may be optional, we need to find a suitable depth format to use
    // Start with the highest precision packed format
    static const std::vector<vk::Format> depthFormats = { vk::Format::eD32SfloatS8Uint, vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint,
                                                          vk::Format::eD16UnormS8Uint, vk::Format::eD16Unorm };

    for (auto& format : depthFormats) {
        vk::FormatProperties formatProps;
        formatProps = physicalDevice.getFormatProperties(format);
        // vk::Format must support depth stencil attachment for optimal tiling
        if (formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            return format;
        }
    }
    throw std::runtime_error("No supported depth format");
}
vks::ExtensionMap vks::DeviceInfo::getExtensions(const vk::PhysicalDevice& handle) {
    vks::ExtensionMap extensions;
    for (const auto ext : handle.enumerateDeviceExtensionProperties()) {
        std::string name = ext.extensionName;
        (extensions).emplace(name, ext);
    }
    return extensions;
}
