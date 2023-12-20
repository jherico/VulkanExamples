/*
* Vulkan Example - 3D texture loading (and generation using perlin noise) example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"
#include <numeric>
#include <future>

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
    float normal[3];
};

// Translation of Ken Perlin's JAVA implementation (http://mrl.nyu.edu/~perlin/noise/)
template <typename T>
class PerlinNoise {
private:
    uint32_t permutations[512];
    static T fade(T t) { return t * t * t * (t * (t * (T)6 - (T)15) + (T)10); }
    static T lerp(T t, T a, T b) { return a + t * (b - a); }
    static T grad(int hash, T x, T y, T z) {
        // Convert LO 4 bits of hash code into 12 gradient directions
        int h = hash & 15;
        T u = h < 8 ? x : y;
        T v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

public:
    PerlinNoise() {
        // Generate random lookup for permutations containing all numbers from 0..255
        std::vector<uint8_t> plookup;
        plookup.resize(256);
        std::iota(plookup.begin(), plookup.end(), 0);
        std::default_random_engine rndEngine(std::random_device{}());
        std::shuffle(plookup.begin(), plookup.end(), rndEngine);

        for (uint32_t i = 0; i < 256; i++) {
            permutations[i] = permutations[256 + i] = plookup[i];
        }
    }

    T noise(T x, T y, T z) {
        // Find unit cube that contains point
        int32_t X = (int32_t)floor(x) & 255;
        int32_t Y = (int32_t)floor(y) & 255;
        int32_t Z = (int32_t)floor(z) & 255;
        // Find relative x,y,z of point in cube
        x -= floor(x);
        y -= floor(y);
        z -= floor(z);

        // Compute fade curves for each of x,y,z
        T u = fade(x);
        T v = fade(y);
        T w = fade(z);

        // Hash coordinates of the 8 cube corners
        uint32_t A = permutations[X] + Y;
        uint32_t AA = permutations[A] + Z;
        uint32_t AB = permutations[A + 1] + Z;
        uint32_t B = permutations[X + 1] + Y;
        uint32_t BA = permutations[B] + Z;
        uint32_t BB = permutations[B + 1] + Z;

        // And add blended results for 8 corners of the cube;
        T res = lerp(w,
                     lerp(v, lerp(u, grad(permutations[AA], x, y, z), grad(permutations[BA], x - 1, y, z)),
                          lerp(u, grad(permutations[AB], x, y - 1, z), grad(permutations[BB], x - 1, y - 1, z))),
                     lerp(v, lerp(u, grad(permutations[AA + 1], x, y, z - 1), grad(permutations[BA + 1], x - 1, y, z - 1)),
                          lerp(u, grad(permutations[AB + 1], x, y - 1, z - 1), grad(permutations[BB + 1], x - 1, y - 1, z - 1))));
        return res;
    }
};

// Fractal noise generator based on perlin noise above
template <typename T>
class FractalNoise {
private:
    PerlinNoise<float> perlinNoise;
    uint32_t octaves;
    T frequency;
    T amplitude;
    T persistence;

public:
    FractalNoise(const PerlinNoise<T>& perlinNoise) {
        this->perlinNoise = perlinNoise;
        octaves = 6;
        persistence = (T)0.5;
    }

    T noise(T x, T y, T z) {
        T sum = 0;
        T frequency = (T)1;
        T amplitude = (T)1;
        T max = (T)0;
        for (uint32_t i = 0; i < octaves; i++) {
            sum += perlinNoise.noise(x * frequency, y * frequency, z * frequency) * amplitude;
            max += amplitude;
            amplitude *= persistence;
            frequency *= (T)2;
        }

        sum = sum / max;
        return (sum + (T)1.0) / (T)2.0;
    }
};

class VulkanExample : public vkx::ExampleBase {
public:
    // Contains all Vulkan objects that are required to store and use a 3D texture
    vks::Image texture;
    vk::Extent3D textureSize;
    vk::DescriptorImageInfo textureDescriptor;

    bool regenerateNoise = true;

    struct {
        vks::model::Model cube;
    } models;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;

    vks::Buffer uniformBufferVS;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 viewPos;
        float depth = 0.0f;
    } uboVS;

    struct {
        vk::Pipeline solid;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        //zoom = -2.5f;
        //rotation = { 0.0f, 15.0f, 0.0f };
        title = "3D textures";
        settings.overlay = true;
        srand((unsigned int)time(NULL));
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        texture.destroy();
        device.destroy(pipelines.solid);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        vertexBuffer.destroy();
        indexBuffer.destroy();
        uniformBufferVS.destroy();
    }

    // Prepare all Vulkan resources for the 3D texture (including descriptors)
    // Does not fill the texture with data
    void prepareNoiseTexture(uint32_t width, uint32_t height, uint32_t depth) {
        // A 3D texture is described as width x height x depth
        textureSize.width = width;
        textureSize.height = height;
        textureSize.depth = depth;
        texture.format = vk::Format::eR8Unorm;

        // Format support check
        // 3D texture support in Vulkan is mandatory (in contrast to OpenGL) so no need to check if it's supported
        auto formatProperties = context.physicalDevice.getFormatProperties(texture.format);
        // Check if format supports transfer
        if (!formatProperties.optimalTilingFeatures && VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
            std::cout << "Error: Device does not support flag TRANSFER_DST for selected texture format!" << std::endl;
            return;
        }
        // Check if GPU supports requested 3D texture dimensions
        uint32_t maxImageDimension3D(context.deviceProperties.limits.maxImageDimension3D);
        if (width > maxImageDimension3D || height > maxImageDimension3D || depth > maxImageDimension3D) {
            std::cout << "Error: Requested texture dimensions is greater than supported 3D texture dimension!" << std::endl;
            return;
        }

        // Create optimal tiled target image
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e3D;
        imageCreateInfo.format = texture.format;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        imageCreateInfo.extent = textureSize;
        // Create image and allocate memory
        texture = context.createImage(imageCreateInfo);

        // Create sampler
        vk::SamplerCreateInfo sampler;
        sampler.magFilter = sampler.minFilter = vk::Filter::eLinear;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        texture.sampler = device.createSampler(sampler);

        // Create image view
        vk::ImageViewCreateInfo view;
        view.image = texture.image;
        view.viewType = vk::ImageViewType::e3D;
        view.format = texture.format;
        view.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        view.subresourceRange.layerCount = 1;
        view.subresourceRange.levelCount = 1;
        texture.view = device.createImageView(view);

        // Fill image descriptor image info to be used descriptor set setup
        textureDescriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        textureDescriptor.imageView = texture.view;
        textureDescriptor.sampler = texture.sampler;
    }

    // Generate randomized noise and upload it to the 3D texture using staging
    void updateNoiseTexture() {
        const uint32_t texMemSize = textureSize.width * textureSize.height * textureSize.depth;

        std::vector<uint8_t> data;
        data.assign(texMemSize, 0);

        // Generate perlin based noise
        std::cout << "Generating " << textureSize.width << " x " << textureSize.height << " x " << textureSize.depth << " noise texture..." << std::endl;

        auto tStart = std::chrono::high_resolution_clock::now();

        PerlinNoise<float> perlinNoise;
        FractalNoise<float> fractalNoise(perlinNoise);

        std::default_random_engine rndEngine(std::random_device{}());
        const int32_t noiseType = rand() % 2;
        const float noiseScale = static_cast<float>(rand() % 10) + 4.0f;

#pragma omp parallel for
        for (int32_t z = 0; z < textureSize.depth; z++) {
            auto zoffset = z * textureSize.width * textureSize.height;
            for (uint32_t y = 0; y < textureSize.height; y++) {
                auto yoffset = y * textureSize.width;
                for (uint32_t x = 0; x < textureSize.width; x++) {
                    float nx = (float)x / (float)textureSize.width;
                    float ny = (float)y / (float)textureSize.height;
                    float nz = (float)z / (float)textureSize.depth;
#define FRACTAL
#ifdef FRACTAL
                    float n = fractalNoise.noise(nx * noiseScale, ny * noiseScale, nz * noiseScale);
#else
                    float n = 20.0 * perlinNoise.noise(nx, ny, nz);
#endif
                    n = n - floor(n);

                    data[x + yoffset + zoffset] = static_cast<uint8_t>(floor(n * 255));
                }
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        std::cout << "Done in " << tDiff << "ms" << std::endl;

        // Create a host-visible staging buffer that contains the raw image data
        // Copy texture data into staging buffer
        vks::Buffer stagingBuffer = context.createStagingBuffer(data);

        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
            // Image barrier for optimal image
            // Optimal image will be used as destination for the copy, so we must transfer from our
            // initial undefined image layout to the transfer destination layout
            context.setImageLayout(copyCmd, texture.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            // Copy 3D noise data to texture

            // Setup buffer copy regions
            vk::BufferImageCopy bufferCopyRegion;
            bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            bufferCopyRegion.imageSubresource.mipLevel = 0;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent = textureSize;
            copyCmd.copyBufferToImage(stagingBuffer.buffer, texture.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);

            // Change texture image layout to shader read after all mip levels have been copied
            context.setImageLayout(copyCmd, texture.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal);
        });

        // Clean up staging resources
        stagingBuffer.destroy();
        regenerateNoise = false;
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        vk::Viewport viewport;
        viewport.width = (float)size.width;
        viewport.height = (float)size.height;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;
        drawCmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor;
        scissor.extent = size;
        drawCmdBuffer.setScissor(0, scissor);

        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);

        VkDeviceSize offsets[1] = { 0 };
        drawCmdBuffer.bindVertexBuffers(0, vertexBuffer.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
    }

    void generateQuad() {
        // Setup vertices for a single uv-mapped quad made from two triangles
        std::vector<Vertex> vertices = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f } },
                                         { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f } },
                                         { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
                                         { { 1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };

        // Setup indices
        std::vector<uint32_t> indices = { 0, 1, 2, 2, 3, 0 };
        indexCount = static_cast<uint32_t>(indices.size());

        // Vertex buffer
        vertexBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertices);

        // Index buffer
        indexBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indices);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };
        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferVS.descriptor },
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textureDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        auto& vertexInputState = builder.vertexInputState;
        // Binding description
        vertexInputState.bindingDescriptions = {
            vk::VertexInputBindingDescription{ 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
        };

        // Attribute descriptions
        vertexInputState.attributeDescriptions = {
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) },     // Location 0 : Position
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv) },         // Location 1 : Texture
            vk::VertexInputAttributeDescription{ 2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal) },  // Location 2 : Normal
        };

        builder.loadShader(getAssetPath() + "shaders/texture3d/texture3d.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/texture3d/texture3d.frag.spv", vk::ShaderStageFlagBits::eFragment);

        pipelines.solid = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformBufferVS = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers(bool viewchanged = true) {
        if (viewchanged) {
            uboVS.projection = camera.matrices.perspective;
            glm::mat4 viewMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -2.5f));
            uboVS.model = viewMatrix * glm::translate(glm::mat4(1.0f), camera.position);
            uboVS.model = glm::rotate(uboVS.model, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
            uboVS.model = glm::rotate(uboVS.model, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
            uboVS.model = glm::rotate(uboVS.model, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
            uboVS.viewPos = glm::vec4(0.0f, 0.0f, 2.5f, 0.0f);
        } else {
            uboVS.depth += frameTimer * 0.15f;
            if (uboVS.depth > 1.0f)
                uboVS.depth = uboVS.depth - 1.0f;
        }

        memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
    }

    void prepare() override {
        ExampleBase::prepare();
        generateQuad();
        prepareUniformBuffers();
        prepareNoiseTexture(256, 256, 256);
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (regenerateNoise) {
            updateNoiseTexture();
        }
        if (!paused) {
            updateUniformBuffers(false);
        }
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (regenerateNoise) {
                ui.text("Generating new noise texture...");
            } else {
                if (ui.button("Generate new texture")) {
                    regenerateNoise = true;
                }
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
