/*
    Vulkan Example - Cascaded shadow mapping for directional light sources
    Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
    This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
    This example implements projective cascaded shadow mapping. This technique splits up the camera frustum into
    multiple frustums with each getting it's own full-res shadow map, implemented as a layered depth-only image.
    The shader then selects the proper shadow map layer depending on what split of the frustum the depth value
    to compare fits into.

    This results in a better shadow map resolution distribution that can be tweaked even further by increasing
    the number of frustum splits.

    A further optimization could be done using a geometry shader to do a single-pass render for the depth map
    cascades instead of multiple passes (geometry shaders are not supported on all target devices).
*/

#include <examples/example.hpp>
#include <shaders/shadowmappingcascade/debugshadowmap.frag.inl>
#include <shaders/shadowmappingcascade/debugshadowmap.vert.inl>
#include <shaders/shadowmappingcascade/depthpass.frag.inl>
#include <shaders/shadowmappingcascade/depthpass.vert.inl>
#include <shaders/shadowmappingcascade/scene.frag.inl>
#include <shaders/shadowmappingcascade/scene.vert.inl>

#if defined(__ANDROID__)
#define SHADOWMAP_DIM 2048
#else
#define SHADOWMAP_DIM 4096
#endif

#define SHADOW_MAP_CASCADE_COUNT 4

class VulkanExample : public vkx::ExampleBase {
public:
    bool displayDepthMap = false;
    int32_t displayDepthMapCascadeIndex = 0;
    bool colorCascades = false;
    bool filterPCF = false;

    float cascadeSplitLambda = 0.95f;

    float zNear = 0.5f;
    float zFar = 48.0f;

    glm::vec3 lightPos = glm::vec3();

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
        vks::model::VERTEX_COMPONENT_NORMAL,
    } };

    std::vector<vks::model::Model> models;

    struct Material {
        vks::texture::Texture2D texture;
        vk::DescriptorSet descriptorSet;
    };
    std::vector<Material> materials;

    struct uniformBuffers {
        vks::Buffer VS;
        vks::Buffer FS;
    } uniformBuffers;

    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::vec3 lightDir;
    } uboVS;

    struct UBOFS {
        float cascadeSplits[4];
        glm::mat4 cascadeViewProjMat[4];
        glm::mat4 inverseViewMat;
        glm::vec3 lightDir;
        float _pad;
        int32_t colorCascades;
    } uboFS;

    vk::PipelineLayout pipelineLayout;
    struct Pipelines {
        vk::Pipeline debugShadowMap;
        vk::Pipeline sceneShadow;
        vk::Pipeline sceneShadowPCF;
    } pipelines;

    struct DescriptorSetLayouts {
        vk::DescriptorSetLayout base;
        vk::DescriptorSetLayout material;
    } descriptorSetLayouts;
    VkDescriptorSet descriptorSet;

    // For simplicity all pipelines use the same push constant block layout
    struct PushConstBlock {
        glm::vec4 position;
        uint32_t cascadeIndex;
    };

    // Resources of the depth map generation pass
    struct DepthPass {
        vk::RenderPass renderPass;
        vk::CommandBuffer commandBuffer;
        vk::PipelineLayout pipelineLayout;
        vk::Pipeline pipeline;
        vks::Buffer uniformBuffer;

        struct UniformBlock {
            std::array<glm::mat4, SHADOW_MAP_CASCADE_COUNT> cascadeViewProjMat;
        } ubo;

        void destroy(const vk::Device& device) {
            device.destroy(renderPass);
            device.destroy(pipelineLayout);
            device.destroy(pipeline);
            uniformBuffer.destroy();
        }
    } depthPass;

    // Layered depth image containing the shadow cascade depths
    struct DepthImage {
        vks::Image image;
        vk::ImageView view;
        vk::Sampler sampler;
    } depth;

    // Contains all resources required for a single shadow map cascade
    struct Cascade {
        vk::Framebuffer frameBuffer;
        vk::DescriptorSet descriptorSet;
        vk::ImageView view;

        float splitDepth;
        glm::mat4 viewProjMatrix;

        void destroy(const vk::Device& device) {
            device.destroy(view);
            device.destroy(frameBuffer);
        }
    };
    std::array<Cascade, SHADOW_MAP_CASCADE_COUNT> cascades;

    VulkanExample() {
        title = "Cascaded shadow mapping";
        timerSpeed *= 0.025f;
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 2.5f;
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, zNear, zFar);
        camera.setPosition(glm::vec3(-0.12f, 1.14f, -2.25f));
        camera.setRotation(glm::vec3(-17.0f, 7.0f, 0.0f));
        settings.overlay = true;
        timer = 0.2f;
    }

    ~VulkanExample() {
        for (auto cascade : cascades) {
            cascade.destroy(device);
        }
        depth.image.destroy();

        device.destroy(pipelines.debugShadowMap);
        device.destroy(pipelines.sceneShadow);
        device.destroy(pipelines.sceneShadowPCF);

        device.destroy(pipelineLayout);

        device.destroy(descriptorSetLayouts.base);
        device.destroy(descriptorSetLayouts.material);

        for (auto& model : models) {
            model.destroy();
        }
        for (auto& material : materials) {
            material.texture.destroy();
        }

        uniformBuffers.VS.destroy();
        uniformBuffers.FS.destroy();
        depthPass.destroy(device);
    }

    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        auto& enabledFeatures = context.enabledFeatures;
        const auto& deviceFeatures = context.deviceInfo.features;
        enabledFeatures.core10.samplerAnisotropy = deviceFeatures.core10.samplerAnisotropy;
        // Depth clamp to avoid near plane clipping
        enabledFeatures.core10.depthClamp = deviceFeatures.core10.depthClamp;
    }

    /*
	    Render the example scene with given command buffer, pipeline layout and dscriptor set
	    Used by the scene rendering and depth pass generation command buffer
	*/
    void renderScene(const vk::CommandBuffer& commandBuffer,
                     const vk::PipelineLayout& pipelineLayout,
                     const vk::DescriptorSet& descriptorSet,
                     uint32_t cascadeIndex = 0) {
        const vk::DeviceSize offsets[1] = { 0 };
        PushConstBlock pushConstBlock = { glm::vec4(0.0f), cascadeIndex };

        std::array<vk::DescriptorSet, 2> sets;
        sets[0] = descriptorSet;

        // Floor
        sets[1] = materials[0].descriptorSet;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, sets, nullptr);
        commandBuffer.pushConstants<PushConstBlock>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstBlock);
        commandBuffer.bindVertexBuffers(0, models[0].vertices.buffer, { 0 });
        commandBuffer.bindIndexBuffer(models[0].indices.buffer, 0, vk::IndexType::eUint32);
        commandBuffer.drawIndexed(models[0].indexCount, 1, 0, 0, 0);

        // Trees
        const std::vector<glm::vec3> positions = {
            glm::vec3(0.0f, 0.0f, 0.0f),    glm::vec3(1.25f, 0.25f, 1.25f),    glm::vec3(-1.25f, -0.2f, 1.25f),
            glm::vec3(1.25f, 0.1f, -1.25f), glm::vec3(-1.25f, -0.25f, -1.25f),
        };

        for (auto position : positions) {
            pushConstBlock.position = glm::vec4(position, 0.0f);
            commandBuffer.pushConstants<PushConstBlock>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstBlock);

            sets[1] = materials[1].descriptorSet;
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, sets, nullptr);
            commandBuffer.bindVertexBuffers(0, models[1].vertices.buffer, { 0 });
            commandBuffer.bindIndexBuffer(models[1].indices.buffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(models[1].indexCount, 1, 0, 0, 0);

            sets[1] = materials[2].descriptorSet;
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, sets, nullptr);
            commandBuffer.bindVertexBuffers(0, models[2].vertices.buffer, { 0 });
            commandBuffer.bindIndexBuffer(models[2].indices.buffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(models[2].indexCount, 1, 0, 0, 0);
        }
    }

    /*
	    Setup resources used by the depth pass
	    The depth image is layered with each layer storing one shadow map cascade
	*/
    void prepareDepthPass() {
        vk::CommandBufferAllocateInfo allocInfo;
        depthPass.commandBuffer = graphicsQueue.allocateCommandBuffers(1)[0];
        // Create a semaphore used to synchronize depth map generation and use

        /*
		    Depth map renderpass
		*/

        vk::AttachmentDescription attachmentDescription;
        attachmentDescription.format = defaultDepthStencilFormat;
        attachmentDescription.loadOp = vk::AttachmentLoadOp::eClear;
        attachmentDescription.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachmentDescription.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachmentDescription.finalLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;

        vk::AttachmentReference depthReference;
        depthReference.attachment = 0;
        depthReference.layout = vk::ImageLayout::eAttachmentOptimal;

        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 0;
        subpass.pDepthStencilAttachment = &depthReference;

        // Use subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eLateFragmentTests;
        dependencies[0].srcAccessMask = vk::AccessFlagBits::eShaderRead;
        dependencies[0].dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eLateFragmentTests;
        dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
        dependencies[1].srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        dependencies[1].dstAccessMask = vk::AccessFlagBits::eShaderRead;
        dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        vk::RenderPassCreateInfo renderPassCreateInfo;
        renderPassCreateInfo.attachmentCount = 1;
        renderPassCreateInfo.pAttachments = &attachmentDescription;
        renderPassCreateInfo.subpassCount = 1;
        renderPassCreateInfo.pSubpasses = &subpass;
        renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassCreateInfo.pDependencies = dependencies.data();

        depthPass.renderPass = device.createRenderPass(renderPassCreateInfo);

        /*
		    Layered depth image and views
		*/

        vk::ImageCreateInfo imageInfo;
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.extent.width = SHADOWMAP_DIM;
        imageInfo.extent.height = SHADOWMAP_DIM;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = SHADOW_MAP_CASCADE_COUNT;
        imageInfo.format = defaultDepthStencilFormat;
        imageInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
        depth.image.create(imageInfo);

        // Full depth map view (all layers)
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.viewType = vk::ImageViewType::e2DArray;
        viewInfo.format = defaultDepthStencilFormat;
        viewInfo.subresourceRange = { vk::ImageAspectFlagBits::eDepth, 0, 1, 0, SHADOW_MAP_CASCADE_COUNT };
        viewInfo.image = depth.image.image;
        depth.view = device.createImageView(viewInfo);

        // One image and framebuffer per cascade
        viewInfo.subresourceRange.layerCount = 1;

        vk::FramebufferCreateInfo framebufferInfo;
        framebufferInfo.renderPass = depthPass.renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.width = SHADOWMAP_DIM;
        framebufferInfo.height = SHADOWMAP_DIM;
        framebufferInfo.layers = 1;

        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            // Image view for this cascade's layer (inside the depth map)
            // This view is used to render to that specific depth image layer
            viewInfo.subresourceRange.baseArrayLayer = i;
            cascades[i].view = device.createImageView(viewInfo);
            // Framebuffer
            framebufferInfo.pAttachments = &cascades[i].view;
            cascades[i].frameBuffer = device.createFramebuffer(framebufferInfo);
        }

        // Shared sampler for cascade deoth reads
        vk::SamplerCreateInfo sampler;
        sampler.magFilter = vk::Filter::eLinear;
        sampler.minFilter = vk::Filter::eLinear;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.maxAnisotropy = 1.0f;
        sampler.minLod = 0.0f;
        sampler.maxLod = 1.0f;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        depth.sampler = device.createSampler(sampler);
    }

    /*
	    Build the command buffer for rendering the depth map cascades
	    Uses multiple passes with each pass rendering the scene to the cascade's depth image layer
	    Could be optimized using a geometry shader (and layered frame buffer) on devices that support geometry shaders
	*/
    void buildDepthPassCommandBuffer() {
        depthPass.commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        vk::Viewport viewport;
        viewport.width = (float)SHADOWMAP_DIM;
        viewport.height = (float)SHADOWMAP_DIM;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;
        depthPass.commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor;
        scissor.extent = vk::Extent2D{ SHADOWMAP_DIM, SHADOWMAP_DIM };
        depthPass.commandBuffer.setScissor(0, scissor);

        vk::ClearValue clearValue;
        clearValue.depthStencil = defaultClearDepth;

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = depthPass.renderPass;
        renderPassBeginInfo.renderArea.extent.width = SHADOWMAP_DIM;
        renderPassBeginInfo.renderArea.extent.height = SHADOWMAP_DIM;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = &clearValue;

        // One pass per cascade
        // The layer that this pass renders too is defined by the cascade's image view (selected via the cascade's decsriptor set)
        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            renderPassBeginInfo.framebuffer = cascades[i].frameBuffer;
            depthPass.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
            depthPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, depthPass.pipeline);
            renderScene(depthPass.commandBuffer, depthPass.pipelineLayout, cascades[i].descriptorSet, i);
            depthPass.commandBuffer.endRenderPass();
        }

        depthPass.commandBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCommandBuffer) override {
        drawCommandBuffer.setViewport(0, vk::Viewport{ 0, 0, (float)size.width, (float)size.height, 0, 1 });
        drawCommandBuffer.setScissor(0, vk::Rect2D{ { 0, 0 }, size });

        vk::DeviceSize offsets[1] = { 0 };

        // Visualize shadow map cascade
        if (displayDepthMap) {
            drawCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, { descriptorSet }, nullptr);
            drawCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debugShadowMap);
            PushConstBlock pushConstBlock = {};
            pushConstBlock.cascadeIndex = displayDepthMapCascadeIndex;
            drawCommandBuffer.pushConstants<PushConstBlock>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstBlock);
            drawCommandBuffer.draw(3, 1, 0, 0);
        }
        drawCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, (filterPCF) ? pipelines.sceneShadowPCF : pipelines.sceneShadow);
        // Render shadowed scene
        renderScene(drawCommandBuffer, pipelineLayout, descriptorSet);
    }

    void loadAssets() override {
        materials.resize(3);
        materials[0].texture.loadFromFile(getAssetPath() + "textures/gridlines.ktx");
        materials[1].texture.loadFromFile(getAssetPath() + "textures/oak_bark.ktx");
        materials[2].texture.loadFromFile(getAssetPath() + "textures/oak_leafs.ktx");

        models.resize(3);
        models[0].loadFromFile(getAssetPath() + "models/terrain_simple.dae", vertexLayout, 1.0f);
        models[1].loadFromFile(getAssetPath() + "models/oak_trunk.dae", vertexLayout, 2.0f);
        models[2].loadFromFile(getAssetPath() + "models/oak_leafs.dae", vertexLayout, 2.0f);
    }

    void setupLayoutsAndDescriptors() {
        // Descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 32 },
            { vk::DescriptorType::eCombinedImageSampler, 32 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 4 + SHADOW_MAP_CASCADE_COUNT, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });

        /*
		    Descriptor set layouts
		*/

        // Shared matrices and samplers
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.base = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });

        // Material texture
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.material = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });

        /*
		    Descriptor sets
		*/
        vk::DescriptorImageInfo depthMapDescriptor{ depth.sampler, depth.view, vk::ImageLayout::eDepthStencilReadOnlyOptimal };
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.base })[0];

        // Scene rendering / debug display
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.VS.descriptor },
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &depthMapDescriptor },
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.FS.descriptor },
        };

        // Per-cascade descriptor sets
        // Each descriptor set represents a single layer of the array texture
        vk::DescriptorImageInfo cascadeImageInfo{ depth.sampler, depth.view, vk::ImageLayout::eDepthStencilReadOnlyOptimal };
        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            cascades[i].descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.base })[0];
            writeDescriptorSets.push_back(
                { cascades[i].descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &depthPass.uniformBuffer.descriptor });
            writeDescriptorSets.push_back({ cascades[i].descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cascadeImageInfo });
            device.updateDescriptorSets(writeDescriptorSets, nullptr);
        }
        // Per-material descriptor sets
        std::vector<vk::DescriptorImageInfo> materialDescriptors;
        materialDescriptors.reserve(materials.size());
        for (auto& material : materials) {
            materialDescriptors.push_back(vk::DescriptorImageInfo{ defaultSampler, material.texture.imageView, vk::ImageLayout::eReadOnlyOptimal });
            material.descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.material })[0];
            writeDescriptorSets.push_back({ material.descriptorSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &materialDescriptors.back() });
        }

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        /*
		    Pipeline layouts
		*/

        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstBlock) };
        std::array<vk::DescriptorSetLayout, 2> setLayouts = { descriptorSetLayouts.base, descriptorSetLayouts.material };
        {
            // Shared pipeline layout (scene and depth map debug display)
            pipelineLayout = device.createPipelineLayout({ {}, (uint32_t)setLayouts.size(), setLayouts.data(), 1, &pushConstantRange });
            // Depth pass pipeline layout
            depthPass.pipelineLayout = device.createPipelineLayout({ {}, (uint32_t)setLayouts.size(), setLayouts.data(), 1, &pushConstantRange });
        }
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // Shadow map cascade debug quad display
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
        builder.loadShader(vkx::shaders::shadowmappingcascade::debugshadowmap::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::shadowmappingcascade::debugshadowmap::frag, vk::ShaderStageFlagBits::eFragment);
        // Empty vertex input state
        pipelines.debugShadowMap = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Vertex bindings and attributes
        vertexLayout.appendVertexLayout(builder.vertexInputState);

        /*
		    Shadow mapped scene rendering
		*/
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        builder.loadShader(vkx::shaders::shadowmappingcascade::scene::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::shadowmappingcascade::scene::frag, vk::ShaderStageFlagBits::eFragment);
        // Use specialization constants to select between horizontal and vertical blur
        uint32_t enablePCF = 0;
        vk::SpecializationMapEntry specializationMapEntry{ 0, 0, sizeof(uint32_t) };
        vk::SpecializationInfo specializationInfo{ 1, &specializationMapEntry, sizeof(uint32_t), &enablePCF };
        builder.shaderStages[1].pSpecializationInfo = &specializationInfo;
        pipelines.sceneShadow = builder.create(context.pipelineCache);
        enablePCF = 1;
        pipelines.sceneShadowPCF = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        /*
		    Depth map generation
		*/
        builder.loadShader(vkx::shaders::shadowmappingcascade::depthpass::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::shadowmappingcascade::depthpass::frag, vk::ShaderStageFlagBits::eFragment);
        // No blend attachment states (no color attachments used)
        builder.colorBlendState.blendAttachmentStates.clear();
        builder.depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
        // Enable depth clamp (if available)
        builder.rasterizationState.depthClampEnable = context.deviceInfo.features.core10.depthClamp;
        builder.layout = depthPass.pipelineLayout;
        builder.renderPass = depthPass.renderPass;
        depthPass.pipeline = builder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Shadow map generation buffer blocks
        depthPass.uniformBuffer = loader.createUniformBuffer(depthPass.ubo);
        // Scene uniform buffer blocks
        uniformBuffers.VS = loader.createUniformBuffer(uboVS);
        uniformBuffers.FS = loader.createUniformBuffer(uboFS);

        updateLight();
        updateUniformBuffers();
    }

    /*
	    Calculate frustum split depths and matrices for the shadow map cascades
	    Based on https://johanmedestrom.wordpress.com/2016/03/18/opengl-cascaded-shadow-maps/
	*/
    void updateCascades() {
        float cascadeSplits[SHADOW_MAP_CASCADE_COUNT];

        float nearClip = camera.getNearClip();
        float farClip = camera.getFarClip();
        float clipRange = farClip - nearClip;

        float minZ = nearClip;
        float maxZ = nearClip + clipRange;

        float range = maxZ - minZ;
        float ratio = maxZ / minZ;

        // Calculate split depths based on view camera furstum
        // Based on method presentd in https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch10.html
        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            float p = (i + 1) / static_cast<float>(SHADOW_MAP_CASCADE_COUNT);
            float log = minZ * std::pow(ratio, p);
            float uniform = minZ + range * p;
            float d = cascadeSplitLambda * (log - uniform) + uniform;
            cascadeSplits[i] = (d - nearClip) / clipRange;
        }

        // Calculate orthographic projection matrix for each cascade
        float lastSplitDist = 0.0;
        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            float splitDist = cascadeSplits[i];

            glm::vec3 frustumCorners[8] = {
                glm::vec3(-1.0f, 1.0f, -1.0f), glm::vec3(1.0f, 1.0f, -1.0f), glm::vec3(1.0f, -1.0f, -1.0f), glm::vec3(-1.0f, -1.0f, -1.0f),
                glm::vec3(-1.0f, 1.0f, 1.0f),  glm::vec3(1.0f, 1.0f, 1.0f),  glm::vec3(1.0f, -1.0f, 1.0f),  glm::vec3(-1.0f, -1.0f, 1.0f),
            };

            // Project frustum corners into world space
            glm::mat4 invCam = glm::inverse(camera.matrices.perspective * camera.matrices.view);
            for (uint32_t i = 0; i < 8; i++) {
                glm::vec4 invCorner = invCam * glm::vec4(frustumCorners[i], 1.0f);
                frustumCorners[i] = invCorner / invCorner.w;
            }

            for (uint32_t i = 0; i < 4; i++) {
                glm::vec3 dist = frustumCorners[i + 4] - frustumCorners[i];
                frustumCorners[i + 4] = frustumCorners[i] + (dist * splitDist);
                frustumCorners[i] = frustumCorners[i] + (dist * lastSplitDist);
            }

            // Get frustum center
            glm::vec3 frustumCenter = glm::vec3(0.0f);
            for (uint32_t i = 0; i < 8; i++) {
                frustumCenter += frustumCorners[i];
            }
            frustumCenter /= 8.0f;

            float radius = 0.0f;
            for (uint32_t i = 0; i < 8; i++) {
                float distance = glm::length(frustumCorners[i] - frustumCenter);
                radius = glm::max(radius, distance);
            }
            radius = std::ceil(radius * 16.0f) / 16.0f;

            glm::vec3 maxExtents = glm::vec3(radius);
            glm::vec3 minExtents = -maxExtents;

            glm::vec3 lightDir = normalize(-lightPos);
            glm::mat4 lightViewMatrix = glm::lookAt(frustumCenter - lightDir * -minExtents.z, frustumCenter, glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 lightOrthoMatrix = glm::ortho(minExtents.x, maxExtents.x, minExtents.y, maxExtents.y, 0.0f, maxExtents.z - minExtents.z);

            // Store split distance and matrix in cascade
            cascades[i].splitDepth = (camera.getNearClip() + splitDist * clipRange) * -1.0f;
            cascades[i].viewProjMatrix = lightOrthoMatrix * lightViewMatrix;

            lastSplitDist = cascadeSplits[i];
        }
    }

    void updateLight() {
        float angle = glm::radians(timer * 360.0f);
        float radius = 20.0f;
        lightPos = glm::vec3(cos(angle) * radius, -radius, sin(angle) * radius);
    }

    void updateUniformBuffers() {
        /*
		    Depth rendering
		*/
        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            depthPass.ubo.cascadeViewProjMat[i] = cascades[i].viewProjMatrix;
        }
        depthPass.uniformBuffer.copy(depthPass.ubo);

        /*
		    Scene rendering
		*/
        uboVS.projection = camera.matrices.perspective;
        uboVS.view = camera.matrices.view;
        uboVS.model = glm::mat4(1.0f);
        uboVS.lightDir = normalize(-lightPos);
        uniformBuffers.VS.copy(uboVS);

        for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
            uboFS.cascadeSplits[i] = cascades[i].splitDepth;
            uboFS.cascadeViewProjMat[i] = cascades[i].viewProjMatrix;
        }
        uboFS.inverseViewMat = glm::inverse(camera.matrices.view);
        uboFS.lightDir = normalize(-lightPos);
        uboFS.colorCascades = colorCascades;
        uniformBuffers.FS.copy(uboFS);
    }

    void preRender() override {
        vks::frame::QueuedCommandBuilder builder{ depthPass.commandBuffer, vkx::RenderStates::OFFSCREEN_PRERENDER, vk::PipelineStageFlagBits2::eAllGraphics };
        queueCommandBuffer(builder);
    }

    void prepare() override {
        ExampleBase::prepare();
        updateLight();
        updateCascades();
        prepareDepthPass();
        prepareUniformBuffers();
        setupLayoutsAndDescriptors();
        preparePipelines();
        buildCommandBuffers();
        buildDepthPassCommandBuffer();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            updateLight();
            updateCascades();
            updateUniformBuffers();
        }
    }

    void viewChanged() override {
        updateCascades();
        updateUniformBuffers();
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.sliderFloat("Split lambda", &cascadeSplitLambda, 0.1f, 1.0f)) {
                updateCascades();
                updateUniformBuffers();
            }
            if (ui.checkBox("Color cascades", &colorCascades)) {
                updateUniformBuffers();
            }
            if (ui.checkBox("Display depth map", &displayDepthMap)) {
                buildCommandBuffers();
            }
            if (displayDepthMap) {
                if (ui.sliderInt("Cascade", &displayDepthMapCascadeIndex, 0, SHADOW_MAP_CASCADE_COUNT - 1)) {
                    buildCommandBuffers();
                }
            }
            if (ui.checkBox("PCF filtering", &filterPCF)) {
                buildCommandBuffers();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
