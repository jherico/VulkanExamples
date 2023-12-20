/*
* Vulkan Example - Omni directional shadows using a dynamic cube map
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanOffscreenExampleBase.hpp>

// Texture properties
#define TEX_DIM 1024
#define TEX_FILTER vk::Filter::eLinear

// Offscreen frame buffer properties
#define FB_DIM TEX_DIM
#define FB_COLOR_FORMAT

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
} };

class VulkanExample : public vkx::OffscreenExampleBase {
public:
    bool displayCubeMap = false;

    float zNear = 0.1f;
    float zFar = 1024.0f;

    struct {
        vk::PipelineVertexInputStateCreateInfo inputState;
        std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    struct {
        vks::model::Model skybox;
        vks::model::Model scene;
    } meshes;

    struct {
        vks::Buffer scene;
        vks::Buffer offscreen;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVSquad;

    glm::vec4 lightPos = glm::vec4(0.0f, -25.0f, 0.0f, 1.0);

    struct {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::vec4 lightPos;
    } uboVSscene;

    struct {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::vec4 lightPos;
    } uboOffscreenVS;

    struct {
        vk::Pipeline scene;
        vk::Pipeline offscreen;
        vk::Pipeline cubeMap;
    } pipelines;

    struct {
        vk::PipelineLayout scene;
        vk::PipelineLayout offscreen;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet scene;
        vk::DescriptorSet offscreen;
    } descriptorSets;

    vk::DescriptorSetLayout descriptorSetLayout;

    const vk::ImageSubresourceRange CUBEMAP_RANGE{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6 };

    vks::Image shadowCubeMap;

    //// vk::Framebuffer for offscreen rendering
    //using FrameBufferAttachment = vkx::CreateImageResult;
    //struct FrameBuffer {
    //    int32_t width, height;
    //    vk::Framebuffer framebuffer;
    //    FrameBufferAttachment color, depth;
    //} offscreenFrameBuf;

    //vk::CommandBuffer offscreen.cmdBuffer;

    VulkanExample() {
        camera.dolly(-175.0f);
        zoomSpeed = 10.0f;
        timerSpeed *= 0.25f;
        camera.setRotation({ -20.5f, -673.0f, 0.0f });
        title = "Vulkan Example - Point light shadows";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        // Cube map
        shadowCubeMap.destroy();

        // Pipelibes
        device.destroyPipeline(pipelines.scene);
        device.destroyPipeline(pipelines.offscreen);
        device.destroyPipeline(pipelines.cubeMap);

        device.destroyPipelineLayout(pipelineLayouts.scene);
        device.destroyPipelineLayout(pipelineLayouts.offscreen);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // Meshes
        meshes.scene.destroy();
        meshes.skybox.destroy();

        // Uniform buffers
        uniformData.offscreen.destroy();
        uniformData.scene.destroy();
    }

    void prepareCubeMap() {
        // 32 bit float format for higher precision
        vk::Format format = vk::Format::eR32Sfloat;
        // Cube map image description
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent.width = TEX_DIM;
        imageCreateInfo.extent.height = TEX_DIM;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 6;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
        imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
        imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageCreateInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;

        shadowCubeMap = context.createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
        context.setImageLayout(shadowCubeMap.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal, CUBEMAP_RANGE);

        vk::ImageSubresourceRange subresourceRange;
        subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 6;

        // Create sampler
        vk::SamplerCreateInfo sampler;
        sampler.magFilter = TEX_FILTER;
        sampler.minFilter = TEX_FILTER;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = vk::SamplerAddressMode::eClampToBorder;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.maxAnisotropy = 0;
        sampler.compareOp = vk::CompareOp::eNever;
        sampler.minLod = 0.0f;
        sampler.maxLod = 0.0f;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        shadowCubeMap.sampler = device.createSampler(sampler);

        // Create image view
        vk::ImageViewCreateInfo view;
        view.viewType = vk::ImageViewType::eCube;
        view.format = format;
        view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        view.subresourceRange.layerCount = 6;
        view.image = shadowCubeMap.image;
        shadowCubeMap.view = device.createImageView(view);
    }

    // Updates a single cube map face
    // Renders the scene with face's view and does
    // a copy from framebuffer to cube face
    // Uses push constants for quick update of
    // view matrix for the current cube map face
    void updateCubeFace(uint32_t faceIndex) {
        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 1.0f });
        clearValues[1].depthStencil = defaultClearDepth;

        vk::RenderPassBeginInfo renderPassBeginInfo;
        // Reuse render pass from example pass
        renderPassBeginInfo.renderPass = offscreen.renderPass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
        renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        // Update view matrix via push constant

        glm::mat4 viewMatrix = glm::mat4();
        switch (faceIndex) {
            case 0:  // POSITIVE_X
                viewMatrix = glm::rotate(viewMatrix, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
                viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                break;
            case 1:  // NEGATIVE_X
                viewMatrix = glm::rotate(viewMatrix, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
                viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                break;
            case 2:  // POSITIVE_Y
                viewMatrix = glm::rotate(viewMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                break;
            case 3:  // NEGATIVE_Y
                viewMatrix = glm::rotate(viewMatrix, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                break;
            case 4:  // POSITIVE_Z
                viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                break;
            case 5:  // NEGATIVE_Z
                viewMatrix = glm::rotate(viewMatrix, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
                break;
        }

        // Change image layout for all cubemap faces to transfer destination
        context.setImageLayout(offscreen.cmdBuffer, offscreen.framebuffers[0].colors[0].image, vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eColorAttachmentOptimal, vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

        // Render scene from cube face's point of view
        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        // Update shader push constant block
        // Contains current face view matrix
        offscreen.cmdBuffer.pushConstants(pipelineLayouts.offscreen, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4), &viewMatrix);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.offscreen, nullptr);
        offscreen.cmdBuffer.bindVertexBuffers(0, meshes.scene.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(meshes.scene.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(meshes.scene.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();

        // Copy region for transfer from framebuffer to cube face
        vk::ImageCopy copyRegion;
        copyRegion.srcSubresource = vk::ImageSubresourceLayers{ vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
        copyRegion.dstSubresource = vk::ImageSubresourceLayers{ vk::ImageAspectFlagBits::eColor, 0, faceIndex, 1 };
        copyRegion.extent = shadowCubeMap.extent;

        // Put image copy into command buffer
        offscreen.cmdBuffer.copyImage(offscreen.framebuffers[0].colors[0].image, vk::ImageLayout::eTransferSrcOptimal, shadowCubeMap.image,
                                      vk::ImageLayout::eTransferDstOptimal, copyRegion);
    }

    // Command buffer for rendering and copying all cube map faces
    void buildOffscreenCommandBuffer() override {
        auto& cmdBuffer = offscreen.cmdBuffer;
        // Create separate command buffer for offscreen
        // rendering
        if (!cmdBuffer) {
            cmdBuffer = context.allocateCommandBuffers(1)[0];
        }
        cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        cmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        // Change image layout for all cubemap faces to transfer destination
        context.setImageLayout(cmdBuffer, shadowCubeMap.image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferDstOptimal, CUBEMAP_RANGE);
        for (uint32_t face = 0; face < 6; ++face) {
            updateCubeFace(face);
        }
        // Change image layout for all cubemap faces to shader read after they have been copied
        context.setImageLayout(cmdBuffer, shadowCubeMap.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, CUBEMAP_RANGE);
        offscreen.cmdBuffer.end();
    }

    //void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {
    //    // Change image layout for all cubemap faces to transfer destination
    //    context.setImageLayout(cmdBuffer, shadowCubeMap.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, CUBEMAP_RANGE);
    //}

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);

        if (displayCubeMap) {
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.cubeMap);
            cmdBuffer.bindVertexBuffers(0, meshes.skybox.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.skybox.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.skybox.indexCount, 1, 0, 0, 0);
        } else {
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.scene);
            cmdBuffer.bindVertexBuffers(0, meshes.scene.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.scene.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.scene.indexCount, 1, 0, 0, 0);
        }
    }

    void loadAssets() override {
        meshes.skybox.loadFromFile(context, getAssetPath() + "models/cube.obj", vertexLayout, 2.0f);
        meshes.scene.loadFromFile(context, getAssetPath() + "models/shadowscene_fire.dae", vertexLayout, 2.0f);
    }

    void setupDescriptorPool() {
        // Example uses three ubos and two image samplers
        std::vector<vk::DescriptorPoolSize> poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
                                                          vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2) };
        descriptorPool = device.createDescriptorPool({ {}, 3, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Shared pipeline layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler (cube map)
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo{ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });

        // 3D scene pipeline layout
        pipelineLayouts.scene = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });

        // Offscreen pipeline layout
        // Push constants for cube map face view matrices
        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4) };
        // Push constant ranges are part of the pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout({ {}, 1, &descriptorSetLayout, 1, &pushConstantRange });
    }

    void setupDescriptorSets() {
        // 3D scene
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        // Offscreen
        descriptorSets.offscreen = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // vk::Image descriptor for the cube map
        vk::DescriptorImageInfo texDescriptor{ shadowCubeMap.sampler, shadowCubeMap.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            vk::WriteDescriptorSet{ descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.scene.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.scene, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
            vk::WriteDescriptorSet{ descriptorSets.offscreen, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.offscreen.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayouts.scene, renderPass };
        // 3D scene pipeline
        // Load shaders
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        builder.loadShader(getAssetPath() + "shaders/shadowmappingomni/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/shadowmappingomni/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelines.scene = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Cube map display pipeline
        builder.loadShader(getAssetPath() + "shaders/shadowmappingomni/cubemapdisplay.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/shadowmappingomni/cubemapdisplay.frag.spv", vk::ShaderStageFlagBits::eFragment);
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        pipelines.cubeMap = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Offscreen pipeline
        builder.loadShader(getAssetPath() + "shaders/shadowmappingomni/offscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/shadowmappingomni/offscreen.frag.spv", vk::ShaderStageFlagBits::eFragment);
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
        builder.layout = pipelineLayouts.offscreen;
        builder.renderPass = offscreen.renderPass;
        pipelines.offscreen = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Offscreen vertex shader uniform buffer block
        uniformData.offscreen = context.createUniformBuffer(uboOffscreenVS);
        // 3D scene
        uniformData.scene = context.createUniformBuffer(uboVSscene);

        updateUniformBufferOffscreen();
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // 3D scene
        uboVSscene.projection = glm::perspective(glm::radians(45.0f), (float)size.width / (float)size.height, zNear, zFar);
        uboVSscene.view = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, displayCubeMap ? 0.0f : camera.position.z));

        glm::mat4 rotM = glm::mat4(1.0f);
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        uboVSscene.model = rotM;
        uboVSscene.lightPos = lightPos;
        uniformData.scene.copy(uboVSscene);
    }

    void updateUniformBufferOffscreen() {
        lightPos.x = sin(glm::radians(timer * 360.0f)) * 1.0f;
        lightPos.z = cos(glm::radians(timer * 360.0f)) * 1.0f;
        uboOffscreenVS.projection = glm::perspective((float)(M_PI / 2.0), 1.0f, zNear, zFar);
        uboOffscreenVS.view = glm::mat4();
        uboOffscreenVS.model = glm::translate(glm::mat4(), glm::vec3(-lightPos.x, -lightPos.y, -lightPos.z));
        uboOffscreenVS.lightPos = lightPos;
        uniformData.offscreen.copy(uboOffscreenVS);
    }

    void prepare() override {
        offscreen.size = glm::uvec2(TEX_DIM);
        offscreen.colorFormats = { vk::Format::eR32Sfloat };
        offscreen.depthFormat = vk::Format::eUndefined;
        offscreen.attachmentUsage = vk::ImageUsageFlagBits::eTransferSrc;
        offscreen.colorFinalLayout = vk::ImageLayout::eTransferSrcOptimal;
        OffscreenExampleBase::prepare();
        prepareUniformBuffers();
        prepareCubeMap();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            updateUniformBufferOffscreen();
            updateUniformBuffers();
        }
    }

    void viewChanged() override {
        updateUniformBufferOffscreen();
        updateUniformBuffers();
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Display shadow cube render target", &displayCubeMap)) {
                buildCommandBuffers();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
