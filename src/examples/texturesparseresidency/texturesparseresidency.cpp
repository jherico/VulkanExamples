/*
* Vulkan Example - Sparse texture residency example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
todos: 
- check sparse binding support on queue
- residencyNonResidentStrict
- meta data
- Run-time image data upload
*/

#include <vulkanExampleBase.h>
#include <heightmap.hpp>

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float normal[3];
    float uv[2];
};

// Virtual texture page as a part of the partially resident texture
// Contains memory bindings, offsets and status information
struct VirtualTexturePage {
    vk::Offset3D offset;
    vk::Extent3D extent;
    vk::SparseImageMemoryBind imageMemoryBind;  // Sparse image memory bind for this page
    vk::DeviceSize size;                        // Page (memory) size in bytes
    uint32_t mipLevel;                          // Mip level that this page belongs to
    uint32_t layer;                             // Array layer that this page belongs to
    uint32_t index;

    // Allocate Vulkan memory for the virtual page
    void allocate(const vk::Device& device, uint32_t memoryTypeIndex) {
        if (imageMemoryBind.memory) {
            //std::cout << "Page " << index << " already allocated" << std::endl;
            return;
        };

        // Sparse image memory binding
        imageMemoryBind.memory = device.allocateMemory({ size, memoryTypeIndex });
        imageMemoryBind.subresource = { vk::ImageAspectFlagBits::eColor, mipLevel, layer };
        imageMemoryBind.extent = extent;
        imageMemoryBind.offset = offset;
    }

    // Release Vulkan memory allocated for this page
    void release(const vk::Device& device) {
        if (imageMemoryBind.memory) {
            device.freeMemory(imageMemoryBind.memory);
            imageMemoryBind.memory = nullptr;
            //std::cout << "Page " << index << " released" << std::endl;
        }
    }
};

// Virtual texture object containing all pages
struct VirtualTexture {
    vk::Device device;
    vk::Image image;                                                // Texture image handle
    vk::BindSparseInfo bindSparseInfo;                              // Sparse queue binding information
    std::vector<VirtualTexturePage> pages;                          // Contains all virtual pages of the texture
    std::vector<vk::SparseImageMemoryBind> sparseImageMemoryBinds;  // Sparse image memory bindings of all memory-backed virtual tables
    std::vector<vk::SparseMemoryBind> opaqueMemoryBinds;            // Sparse ópaque memory bindings for the mip tail (if present)
    vk::SparseImageMemoryBindInfo imageMemoryBindInfo;              // Sparse image memory bind info
    vk::SparseImageOpaqueMemoryBindInfo opaqueMemoryBindInfo;       // Sparse image opaque memory bind info (mip tail)
    uint32_t mipTailStart;                                          // First mip level in mip tail

    VirtualTexturePage* addPage(vk::Offset3D offset, vk::Extent3D extent, const vk::DeviceSize size, const uint32_t mipLevel, uint32_t layer) {
        VirtualTexturePage newPage;
        newPage.offset = offset;
        newPage.extent = extent;
        newPage.size = size;
        newPage.mipLevel = mipLevel;
        newPage.layer = layer;
        newPage.index = static_cast<uint32_t>(pages.size());
        newPage.imageMemoryBind.offset = offset;
        newPage.imageMemoryBind.extent = extent;
        pages.push_back(newPage);
        return &pages.back();
    }

    // Call before sparse binding to update memory bind list etc.
    void updateSparseBindInfo() {
        // Update list of memory-backed sparse image memory binds
        sparseImageMemoryBinds.resize(pages.size());
        uint32_t index = 0;
        for (auto page : pages) {
            sparseImageMemoryBinds[index] = page.imageMemoryBind;
            index++;
        }
        // Update sparse bind info
        bindSparseInfo = vk::BindSparseInfo{};
        // todo: Semaphore for queue submission
        // bindSparseInfo.signalSemaphoreCount = 1;
        // bindSparseInfo.pSignalSemaphores = &bindSparseSemaphore;

        // Image memory binds
        imageMemoryBindInfo.image = image;
        imageMemoryBindInfo.bindCount = static_cast<uint32_t>(sparseImageMemoryBinds.size());
        imageMemoryBindInfo.pBinds = sparseImageMemoryBinds.data();
        bindSparseInfo.imageBindCount = (imageMemoryBindInfo.bindCount > 0) ? 1 : 0;
        bindSparseInfo.pImageBinds = &imageMemoryBindInfo;

        // Opaque image memory binds (mip tail)
        opaqueMemoryBindInfo.image = image;
        opaqueMemoryBindInfo.bindCount = static_cast<uint32_t>(opaqueMemoryBinds.size());
        opaqueMemoryBindInfo.pBinds = opaqueMemoryBinds.data();
        bindSparseInfo.imageOpaqueBindCount = (opaqueMemoryBindInfo.bindCount > 0) ? 1 : 0;
        bindSparseInfo.pImageOpaqueBinds = &opaqueMemoryBindInfo;
    }

    // Release all Vulkan resources
    void destroy() {
        for (auto page : pages) {
            page.release(device);
        }
        for (auto bind : opaqueMemoryBinds) {
            device.freeMemory(bind.memory);
        }
    }
};

uint32_t memoryTypeIndex;
int32_t lastFilledMip = 0;

class VulkanExample : public vkx::ExampleBase {
public:
    //todo: comments
    struct SparseTexture : VirtualTexture {
        vk::Sampler sampler;
        vk::ImageLayout imageLayout;
        vk::ImageView view;
        vk::DescriptorImageInfo descriptor;
        vk::Format format;
        uint32_t width, height;
        uint32_t mipLevels;
        uint32_t layerCount;
    } texture;

    struct {
        vks::texture::Texture2D source;
    } textures;

    vkx::HeightMap heightMap;

    uint32_t indexCount{ 0 };

    vks::Buffer uniformBufferVS;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 viewPos;
        float lodBias = 0.0f;
    } uboVS;

    struct {
        vk::Pipeline solid;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    //todo: comment
    vk::Semaphore bindSparseSemaphore;

    VulkanExample() {
        title = "Sparse texture residency";
        std::cout.imbue(std::locale(""));
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 50.0f;
#ifndef __ANDROID__
        camera.rotationSpeed = 0.25f;
#endif
        camera.position = { 84.5f, 40.5f, 225.0f };
        camera.setRotation(glm::vec3(-8.5f, -200.0f, 0.0f));
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 1024.0f);
        settings.overlay = true;
        // Device features to be enabled for this example
        enabledFeatures.shaderResourceResidency = VK_TRUE;
        enabledFeatures.shaderResourceMinLod = VK_TRUE;
    }

    ~VulkanExample() {
        textures.source.destroy();
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        heightMap.destroy();

        destroyTextureImage(texture);
        device.destroy(bindSparseSemaphore);
        device.destroy(pipelines.solid);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        uniformBufferVS.destroy();
    }

    virtual void getEnabledFeatures() override {
        if (deviceFeatures.sparseBinding && deviceFeatures.sparseResidencyImage2D) {
            enabledFeatures.sparseBinding = VK_TRUE;
            enabledFeatures.sparseResidencyImage2D = VK_TRUE;
        } else {
            std::cout << "Sparse binding not supported" << std::endl;
        }
    }

    static glm::uvec3 alignedDivision(const vk::Extent3D& extent, const vk::Extent3D& granularity) {
        glm::uvec3 res;
        res.x = extent.width / granularity.width + ((extent.width % granularity.width) ? 1u : 0u);
        res.y = extent.height / granularity.height + ((extent.height % granularity.height) ? 1u : 0u);
        res.z = extent.depth / granularity.depth + ((extent.depth % granularity.depth) ? 1u : 0u);
        return res;
    }

    void prepareSparseTexture(uint32_t width, uint32_t height, uint32_t layerCount, vk::Format format) {
        texture.device = device;
        texture.width = width;
        texture.height = height;
        texture.mipLevels = (uint32_t)floor(log2(std::max(width, height))) + 1;
        texture.layerCount = layerCount;
        texture.format = format;

        // Get device properites for the requested texture format
        vk::FormatProperties formatProperties = context.physicalDevice.getFormatProperties(format);

        // Get sparse image properties
        std::vector<vk::SparseImageFormatProperties> sparseProperties =
            context.physicalDevice.getSparseImageFormatProperties(format, vk::ImageType::e2D, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eSampled,
                                                                  vk::ImageTiling::eOptimal);
        // Sparse properties count for the desired format
        // Check if sparse is supported for this format
        if (sparseProperties.empty()) {
            std::cout << "Error: Requested format does not support sparse features!" << std::endl;
            return;
        }

        std::cout << "Sparse image format properties: " << sparseProperties.size() << std::endl;
        for (auto props : sparseProperties) {
            std::cout << "\t Image granularity: w = " << props.imageGranularity.width << " h = " << props.imageGranularity.height
                      << " d = " << props.imageGranularity.depth << std::endl;
            std::cout << "\t Aspect mask: " << vk::to_string(props.aspectMask) << std::endl;
            std::cout << "\t Flags: " << vk::to_string(props.flags) << std::endl;
        }

        // Create sparse image
        vk::ImageCreateInfo sparseImageCreateInfo;
        sparseImageCreateInfo.imageType = vk::ImageType::e2D;
        sparseImageCreateInfo.format = texture.format;
        sparseImageCreateInfo.mipLevels = texture.mipLevels;
        sparseImageCreateInfo.arrayLayers = texture.layerCount;
        sparseImageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        sparseImageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        sparseImageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        sparseImageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
        sparseImageCreateInfo.extent = vk::Extent3D{ texture.width, texture.height, 1 };
        sparseImageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
        sparseImageCreateInfo.flags = vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency;
        texture.image = device.createImage(sparseImageCreateInfo);

        vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, 0, texture.mipLevels, 0, 1 };
        context.setImageLayout(texture.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, range);

        // Sparse image memory requirement counts
        vk::MemoryRequirements sparseImageMemoryReqs = device.getImageMemoryRequirements(texture.image);

        std::cout << "Image memory requirements:" << std::endl;
        std::cout << "\t Size: " << sparseImageMemoryReqs.size << std::endl;
        std::cout << "\t Alignment: " << sparseImageMemoryReqs.alignment << std::endl;

        // Check requested image size against hardware sparse limit
        if (sparseImageMemoryReqs.size > context.deviceProperties.limits.sparseAddressSpaceSize) {
            std::cout << "Error: Requested sparse image size exceeds supportes sparse address space size!" << std::endl;
            return;
        };

        // Get sparse memory requirements
        std::vector<vk::SparseImageMemoryRequirements> sparseMemoryReqs = device.getImageSparseMemoryRequirements(texture.image);
        if (sparseMemoryReqs.empty()) {
            std::cout << "Error: No memory requirements for the sparse image!" << std::endl;
            return;
        }

        std::cout << "Sparse image memory requirements: " << sparseMemoryReqs.size() << std::endl;
        for (auto reqs : sparseMemoryReqs) {
            std::cout << "\t Image granularity: w = " << reqs.formatProperties.imageGranularity.width
                      << " h = " << reqs.formatProperties.imageGranularity.height << " d = " << reqs.formatProperties.imageGranularity.depth << std::endl;
            std::cout << "\t Mip tail first LOD: " << reqs.imageMipTailFirstLod << std::endl;
            std::cout << "\t Mip tail size: " << reqs.imageMipTailSize << std::endl;
            std::cout << "\t Mip tail offset: " << reqs.imageMipTailOffset << std::endl;
            std::cout << "\t Mip tail stride: " << reqs.imageMipTailStride << std::endl;
            //todo:multiple reqs
            texture.mipTailStart = reqs.imageMipTailFirstLod;
        }

        lastFilledMip = texture.mipTailStart - 1;

        // Get sparse image requirements for the color aspect
        vk::SparseImageMemoryRequirements sparseMemoryReq;
        bool colorAspectFound = false;
        for (auto reqs : sparseMemoryReqs) {
            if (reqs.formatProperties.aspectMask & vk::ImageAspectFlagBits::eColor) {
                sparseMemoryReq = reqs;
                colorAspectFound = true;
                break;
            }
        }
        if (!colorAspectFound) {
            std::cout << "Error: Could not find sparse image memory requirements for color aspect bit!" << std::endl;
            return;
        }

        // todo:
        // Calculate number of required sparse memory bindings by alignment
        assert((sparseImageMemoryReqs.size % sparseImageMemoryReqs.alignment) == 0);
        memoryTypeIndex = context.getMemoryType(sparseImageMemoryReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Get sparse bindings
        uint32_t sparseBindsCount = static_cast<uint32_t>(sparseImageMemoryReqs.size / sparseImageMemoryReqs.alignment);
        std::vector<vk::SparseMemoryBind> sparseMemoryBinds(sparseBindsCount);

        // Check if the format has a single mip tail for all layers or one mip tail for each layer
        // The mip tail contains all mip levels > sparseMemoryReq.imageMipTailFirstLod
        bool singleMipTail = false;
        if (sparseMemoryReq.formatProperties.flags & vk::SparseImageFormatFlagBits::eSingleMiptail) {
            singleMipTail = true;
        }

        // Sparse bindings for each mip level of all layers outside of the mip tail
        for (uint32_t layer = 0; layer < texture.layerCount; layer++) {
            // sparseMemoryReq.imageMipTailFirstLod is the first mip level that's stored inside the mip tail
            for (uint32_t mipLevel = 0; mipLevel < sparseMemoryReq.imageMipTailFirstLod; mipLevel++) {
                vk::Extent3D extent;
                extent.width = std::max(sparseImageCreateInfo.extent.width >> mipLevel, 1u);
                extent.height = std::max(sparseImageCreateInfo.extent.height >> mipLevel, 1u);
                extent.depth = std::max(sparseImageCreateInfo.extent.depth >> mipLevel, 1u);

                vk::ImageSubresource subResource{};
                subResource.aspectMask = vk::ImageAspectFlagBits::eColor;
                subResource.mipLevel = mipLevel;
                subResource.arrayLayer = layer;

                // Aligned sizes by image granularity
                vk::Extent3D imageGranularity = sparseMemoryReq.formatProperties.imageGranularity;
                glm::uvec3 sparseBindCounts = alignedDivision(extent, imageGranularity);
                glm::uvec3 lastBlockExtent;
                lastBlockExtent.x = (extent.width % imageGranularity.width) ? extent.width % imageGranularity.width : imageGranularity.width;
                lastBlockExtent.y = (extent.height % imageGranularity.height) ? extent.height % imageGranularity.height : imageGranularity.height;
                lastBlockExtent.z = (extent.depth % imageGranularity.depth) ? extent.depth % imageGranularity.depth : imageGranularity.depth;

                // Alllocate memory for some blocks
                uint32_t index = 0;
                for (uint32_t z = 0; z < sparseBindCounts.z; z++) {
                    for (uint32_t y = 0; y < sparseBindCounts.y; y++) {
                        for (uint32_t x = 0; x < sparseBindCounts.x; x++) {
                            // Offset
                            VkOffset3D offset;
                            offset.x = x * imageGranularity.width;
                            offset.y = y * imageGranularity.height;
                            offset.z = z * imageGranularity.depth;
                            // Size of the page
                            VkExtent3D extent;
                            extent.width = (x == sparseBindCounts.x - 1) ? lastBlockExtent.x : imageGranularity.width;
                            extent.height = (y == sparseBindCounts.y - 1) ? lastBlockExtent.y : imageGranularity.height;
                            extent.depth = (z == sparseBindCounts.z - 1) ? lastBlockExtent.z : imageGranularity.depth;

                            // Add new virtual page
                            VirtualTexturePage* newPage = texture.addPage(offset, extent, sparseImageMemoryReqs.alignment, mipLevel, layer);
                            newPage->imageMemoryBind.subresource = subResource;

                            if ((x % 2 == 1) || (y % 2 == 1)) {
                                // Allocate memory for this virtual page
                                //newPage->allocate(device, memoryTypeIndex);
                            }

                            index++;
                        }
                    }
                }
            }

            // Check if format has one mip tail per layer
            if ((!singleMipTail) && (sparseMemoryReq.imageMipTailFirstLod < texture.mipLevels)) {
                // Allocate memory for the mip tail
                vk::DeviceMemory deviceMemory = device.allocateMemory({ sparseMemoryReq.imageMipTailSize, memoryTypeIndex });

                // (Opaque) sparse memory binding
                vk::SparseMemoryBind sparseMemoryBind;
                sparseMemoryBind.resourceOffset = sparseMemoryReq.imageMipTailOffset + layer * sparseMemoryReq.imageMipTailStride;
                sparseMemoryBind.size = sparseMemoryReq.imageMipTailSize;
                sparseMemoryBind.memory = deviceMemory;
                texture.opaqueMemoryBinds.push_back(sparseMemoryBind);
            }
        }  // end layers and mips

        std::cout << "Texture info:" << std::endl;
        std::cout << "\tDim: " << texture.width << " x " << texture.height << std::endl;
        std::cout << "\tVirtual pages: " << texture.pages.size() << std::endl;

        // Check if format has one mip tail for all layers
        if ((sparseMemoryReq.formatProperties.flags & vk::SparseImageFormatFlagBits::eSingleMiptail) &&
            (sparseMemoryReq.imageMipTailFirstLod < texture.mipLevels)) {
            // Allocate memory for the mip tail
            vk::DeviceMemory deviceMemory = device.allocateMemory({ sparseMemoryReq.imageMipTailSize, memoryTypeIndex });

            // (Opaque) sparse memory binding
            vk::SparseMemoryBind sparseMemoryBind;
            sparseMemoryBind.resourceOffset = sparseMemoryReq.imageMipTailOffset;
            sparseMemoryBind.size = sparseMemoryReq.imageMipTailSize;
            sparseMemoryBind.memory = deviceMemory;

            texture.opaqueMemoryBinds.push_back(sparseMemoryBind);
        }


        // populate the mip tail
        {
            std::vector<vk::ImageBlit> imageBlits;
            for (uint32_t mipLevel = texture.mipTailStart; mipLevel < texture.mipLevels; ++mipLevel) {
                vk::Extent3D extent;
                extent.width = std::max(textures.source.extent.width >> mipLevel, 1u);
                extent.height = std::max(textures.source.extent.height >> mipLevel, 1u);
                extent.depth = std::max(textures.source.extent.depth >> mipLevel, 1u);

                // Image blit
                vk::ImageBlit blit;
                // Source
                blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                blit.srcSubresource.baseArrayLayer = 0;
                blit.srcSubresource.layerCount = 1;
                blit.srcSubresource.mipLevel = 0;
                blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
                blit.srcOffsets[1] = vk::Offset3D{ static_cast<int32_t>(textures.source.extent.width), static_cast<int32_t>(textures.source.extent.height), 1 };
                // Dest
                blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                blit.dstSubresource.baseArrayLayer = 0;
                blit.dstSubresource.layerCount = 1;
                blit.dstSubresource.mipLevel = mipLevel;
                blit.dstOffsets[0].x = 0;
                blit.dstOffsets[0].y = 0;
                blit.dstOffsets[0].z = 0;
                blit.dstOffsets[1].x = extent.width;
                blit.dstOffsets[1].y = extent.height;
                blit.dstOffsets[1].z = extent.depth;
                imageBlits.push_back(blit);
            }

            // Issue blit commands
            if (!imageBlits.empty()) {
                vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, (uint32_t)texture.mipTailStart, texture.mipLevels - texture.mipTailStart, 0, 1 };
                context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
                    context.setImageLayout(copyCmd, texture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal, range);
                    copyCmd.blitImage(                                                //
                        textures.source.image, vk::ImageLayout::eTransferSrcOptimal,  //
                        texture.image, vk::ImageLayout::eTransferDstOptimal,          //
                        imageBlits, vk::Filter::eLinear);
                    context.setImageLayout(copyCmd, texture.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, range);
                });
            }
        }

        // Create signal semaphore for sparse binding
        bindSparseSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});

        // Prepare bind sparse info for reuse in queue submission
        texture.updateSparseBindInfo();

        // Bind to queue
        // todo: in draw?
        queue.bindSparse(texture.bindSparseInfo, nullptr);
        //todo: use sparse bind semaphore
        queue.waitIdle();

        // Create sampler
        vk::SamplerCreateInfo sampler;
        sampler.magFilter = vk::Filter::eLinear;
        sampler.minFilter = vk::Filter::eLinear;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = vk::SamplerAddressMode::eRepeat;
        sampler.addressModeV = vk::SamplerAddressMode::eRepeat;
        sampler.addressModeW = vk::SamplerAddressMode::eRepeat;
        sampler.mipLodBias = 0.0f;
        sampler.compareOp = vk::CompareOp::eNever;
        sampler.minLod = 0.0f;
        sampler.maxLod = static_cast<float>(texture.mipLevels);
        sampler.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
        sampler.anisotropyEnable = false;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        texture.sampler = device.createSampler(sampler);

        // Create image view
        vk::ImageViewCreateInfo view;
        view.viewType = vk::ImageViewType::e2D;
        view.format = format;
        view.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        view.subresourceRange.baseMipLevel = 0;
        view.subresourceRange.baseArrayLayer = 0;
        view.subresourceRange.layerCount = 1;
        view.subresourceRange.levelCount = texture.mipLevels;
        view.image = texture.image;
        texture.view = device.createImageView(view);

        // Fill image descriptor image info that can be used during the descriptor set setup
        texture.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        texture.descriptor.imageView = texture.view;
        texture.descriptor.sampler = texture.sampler;

        // Fill smallest (non-tail) mip map leve
        fillVirtualTexture(lastFilledMip);
    }

    // Free all Vulkan resources used a texture object
    void destroyTextureImage(SparseTexture texture) {
        vkDestroyImageView(device, texture.view, nullptr);
        vkDestroyImage(device, texture.image, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        texture.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        drawCmdBuffer.bindVertexBuffers(0, heightMap.vertexBuffer.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(heightMap.indexBuffer.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(heightMap.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        textures.source.loadFromFile(context, getAssetPath() + "textures/ground_dry_bc3_unorm.ktx", vk::Format::eBc3UnormBlock,
                                     vk::ImageUsageFlagBits::eTransferSrc, vk::ImageLayout::eTransferSrcOptimal);
        // Generate a terrain quad patch for feeding to the tessellation control shader
        heightMap.loadFromFile(context, getAssetPath() + "textures/terrain_heightmap_r16.ktx", 128, glm::vec3(2.0f, 48.0f, 2.0f),
                               vkx::HeightMap::topologyTriangles);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferVS.descriptor },
            vk::WriteDescriptorSet{ descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texture.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.vertexInputState.bindingDescriptions = {
            vk::VertexInputBindingDescription{ 0, sizeof(Vertex) },
        };
        builder.vertexInputState.attributeDescriptions = {
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) },
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal) },
            vk::VertexInputAttributeDescription{ 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv) },
        };
        builder.loadShader(getAssetPath() + "shaders/texturesparseresidency/sparseresidency.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/texturesparseresidency/sparseresidency.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformBufferVS = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uboVS.viewPos = glm::vec4(0.0f, 0.0f, -10.0f, 0.0f);
        memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
    }

    void prepare() override {
        ExampleBase::prepare();
        // Check if the GPU supports sparse residency for 2D images
        if (!context.deviceFeatures.sparseResidencyImage2D) {
            throw std::runtime_error("Device does not support sparse residency for 2D images!");
        }
        prepareUniformBuffers();
        // Create a virtual texture with max. possible dimension (does not take up any VRAM yet)
        prepareSparseTexture(8192, 8192, 1, vk::Format::eR8G8B8A8Unorm);
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    // Clear all pages of the virtual texture
    // todo: just for testing
    void flushVirtualTexture() {
        queue.waitIdle();
        device.waitIdle();
        for (auto& page : texture.pages) {
            if (page.mipLevel < texture.mipTailStart - 1) {
                page.release(device);
            }
        }
        texture.updateSparseBindInfo();
        queue.bindSparse(texture.bindSparseInfo, nullptr);
        //todo: use sparse bind semaphore
        queue.waitIdle();
        device.waitIdle();
        lastFilledMip = texture.mipTailStart - 2;
    }

    // Fill a complete mip level
    void fillVirtualTexture(int32_t& mipLevel) {
        device.waitIdle();
        std::vector<vk::ImageBlit> imageBlits;
        for (auto& page : texture.pages) {
            if ((page.mipLevel == mipLevel) && /*(rndDist(rndEngine) < 0.5f) &&*/ !page.imageMemoryBind.memory) {
                // Allocate page memory
                page.allocate(device, memoryTypeIndex);

                // Current mip level scaling
                uint32_t scale = texture.width / (texture.width >> page.mipLevel);

                for (uint32_t x = 0; x < scale; x++) {
                    for (uint32_t y = 0; y < scale; y++) {
                        // Image blit
                        vk::ImageBlit blit;
                        // Source
                        blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                        blit.srcSubresource.baseArrayLayer = 0;
                        blit.srcSubresource.layerCount = 1;
                        blit.srcSubresource.mipLevel = 0;
                        blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
                        blit.srcOffsets[1] = vk::Offset3D{ static_cast<int32_t>(textures.source.extent.width), static_cast<int32_t>(textures.source.extent.height), 1 };
                        // Dest
                        blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                        blit.dstSubresource.baseArrayLayer = 0;
                        blit.dstSubresource.layerCount = 1;
                        blit.dstSubresource.mipLevel = page.mipLevel;
                        blit.dstOffsets[0].x = static_cast<int32_t>(page.offset.x + x * 128 / scale);
                        blit.dstOffsets[0].y = static_cast<int32_t>(page.offset.y + y * 128 / scale);
                        blit.dstOffsets[0].z = 0;
                        blit.dstOffsets[1].x = static_cast<int32_t>(blit.dstOffsets[0].x + page.extent.width / scale);
                        blit.dstOffsets[1].y = static_cast<int32_t>(blit.dstOffsets[0].y + page.extent.height / scale);
                        blit.dstOffsets[1].z = 1;

                        imageBlits.push_back(blit);
                    }
                }
            }
        }

        // Update sparse queue binding
        texture.updateSparseBindInfo();
        queue.bindSparse(texture.bindSparseInfo, nullptr);
        //todo: use sparse bind semaphore
        queue.waitIdle();

        // Issue blit commands
        if (imageBlits.size() > 0) {
            auto tStart = std::chrono::high_resolution_clock::now();
            vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, (uint32_t)mipLevel, 1, 0, 1 };
            context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
                context.setImageLayout(copyCmd, texture.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal, range);
                copyCmd.blitImage(                                                //
                    textures.source.image, vk::ImageLayout::eTransferSrcOptimal,  //
                    texture.image, vk::ImageLayout::eTransferDstOptimal,          //
                    imageBlits, vk::Filter::eLinear);
                context.setImageLayout(copyCmd, texture.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, range);
            });
            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            std::cout << "Image blits took " << tDiff << " ms" << std::endl;
        }

        queue.waitIdle();
        mipLevel--;
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.sliderFloat("LOD bias", &uboVS.lodBias, 0.0f, (float)texture.mipLevels)) {
                updateUniformBuffers();
            }
            ui.text("Last filled mip level: %d", lastFilledMip);
            if (lastFilledMip > 0) {
                if (ui.button("Fill next mip level")) {
                    fillVirtualTexture(lastFilledMip);
                }
            }
            if (ui.button("Flush virtual texture")) {
                flushVirtualTexture();
            }
        }
        if (ui.header("Statistics")) {
            uint32_t respages = 0;
            std::for_each(texture.pages.begin(), texture.pages.end(),
                          [&respages](VirtualTexturePage page) { respages += (page.imageMemoryBind.memory) ? 1 : 0; });
            ui.text("Resident pages: %d of %d", respages, static_cast<uint32_t>(texture.pages.size()));
        }
    }
};

VULKAN_EXAMPLE_MAIN()
