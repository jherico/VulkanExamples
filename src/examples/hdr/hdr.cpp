/*
* Vulkan Example - HDR
*
* Note: Requires the separate asset pack (see data/README.md)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase {
public:
    bool bloom = true;
    bool displaySkybox = true;

    // Vertex layout for the models

    struct {
        vks::texture::TextureCubeMap envmap;
    } textures;

    struct Models {
        vks::model::VertexLayout vertexLayout{ {
            vks::model::VERTEX_COMPONENT_POSITION,
            vks::model::VERTEX_COMPONENT_NORMAL,
            vks::model::VERTEX_COMPONENT_UV,
        } };
        vks::model::Model skybox;
        std::vector<vks::model::Model> objects;
        int32_t objectIndex = 1;
    } models;

    struct {
        vks::Buffer matrices;
        vks::Buffer params;
    } uniformBuffers;

    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 modelview;
    } uboVS;

    struct UBOParams {
        float exposure = 1.0f;
    } uboParams;

    struct {
        vk::Pipeline skybox;
        vk::Pipeline reflect;
        vk::Pipeline composition;
        vk::Pipeline bloom[2];
    } pipelines;

    struct {
        vk::PipelineLayout models;
        vk::PipelineLayout composition;
        vk::PipelineLayout bloomFilter;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet object;
        vk::DescriptorSet skybox;
        vk::DescriptorSet composition;
        vk::DescriptorSet bloomFilter;
    } descriptorSets;

    struct {
        vk::DescriptorSetLayout models;
        vk::DescriptorSetLayout composition;
        vk::DescriptorSetLayout bloomFilter;
    } descriptorSetLayouts;

    struct {
        vk::Extent2D extent;
        vk::Framebuffer frameBuffer;
        vks::Image color[2];
        vks::Image depth;
        vk::RenderPass renderPass;
        vk::Sampler sampler;
        vk::CommandBuffer cmdBuffer;
        vk::Semaphore semaphore;
    } offscreen;

    struct {
        vk::Extent2D extent;
        VkFramebuffer frameBuffer;
        vks::Image color[1];
        VkRenderPass renderPass;
        VkSampler sampler;
    } filterPass;

    std::vector<std::string> objectNames;

    VulkanExample() {
        title = "Hight dynamic range rendering";
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -4.0f));
        camera.setRotation(glm::vec3(0.0f, 180.0f, 0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroyPipeline(pipelines.skybox);
        device.destroyPipeline(pipelines.reflect);
        device.destroyPipeline(pipelines.composition);
        device.destroyPipeline(pipelines.bloom[0]);
        device.destroyPipeline(pipelines.bloom[1]);

        device.destroyPipelineLayout(pipelineLayouts.models);
        device.destroyPipelineLayout(pipelineLayouts.composition);
        device.destroyPipelineLayout(pipelineLayouts.bloomFilter);

        device.destroyDescriptorSetLayout(descriptorSetLayouts.models);
        device.destroyDescriptorSetLayout(descriptorSetLayouts.composition);
        device.destroyDescriptorSetLayout(descriptorSetLayouts.bloomFilter);

        device.destroySemaphore(offscreen.semaphore);

        device.destroyRenderPass(offscreen.renderPass);
        device.destroyRenderPass(filterPass.renderPass);

        device.destroyFramebuffer(offscreen.frameBuffer);
        device.destroyFramebuffer(filterPass.frameBuffer);

        vkDestroySampler(device, offscreen.sampler, nullptr);
        vkDestroySampler(device, filterPass.sampler, nullptr);

        offscreen.depth.destroy();
        offscreen.color[0].destroy();
        offscreen.color[1].destroy();

        filterPass.color[0].destroy();

        for (auto& model : models.objects) {
            model.destroy();
        }
        models.skybox.destroy();
        uniformBuffers.matrices.destroy();
        uniformBuffers.params.destroy();
        textures.envmap.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) override {
        // Bloom filter
        clearValues = {
            vks::util::clearColor(glm::vec4(0.0f)),
        };
        renderPassBeginInfo.framebuffer = filterPass.frameBuffer;
        renderPassBeginInfo.renderPass = filterPass.renderPass;
        renderPassBeginInfo.renderArea.extent = filterPass.extent;
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.clearValueCount = 1;

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        commandBuffer.setViewport(0, vks::util::viewport(filterPass.extent));
        commandBuffer.setScissor(0, vks::util::rect2D(filterPass.extent));
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.bloomFilter, 0, descriptorSets.bloomFilter, nullptr);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.bloom[1]);
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRenderPass();

        // Final composition
        clearValues = {
            vks::util::clearColor(glm::vec4(0.0f)),
            defaultClearDepth,
        };
        renderPassBeginInfo.framebuffer = framebuffers[currentBuffer];
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent = size;
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.clearValueCount = 2;

        // Scene
        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        commandBuffer.setViewport(0, viewport());
        commandBuffer.setScissor(0, scissor());
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, nullptr);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
        commandBuffer.draw(3, 1, 0, 0);

        // Bloom
        if (bloom) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.bloom[0]);
            commandBuffer.draw(3, 1, 0, 0);
        }

        commandBuffer.endRenderPass();
    }

    void buildCommandBuffers() override {
        // Destroy and recreate command buffers if already present
        allocateCommandBuffers();

        for (int32_t i = 0; i < commandBuffers.size(); ++i) {
            currentBuffer = i;
            commandBuffers[i].begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
            updateDrawCommandBuffer(commandBuffers[i]);
            commandBuffers[i].end();
        }
    }

    vks::Image createAttachment(vk::Format format, vk::ImageUsageFlags usage) {
        vk::ImageAspectFlags aspectMask;
        vk::ImageLayout imageLayout;

        if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eColor;
            imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }
        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        }

        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent.width = offscreen.extent.width;
        imageCreateInfo.extent.height = offscreen.extent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.usage = usage | vk::ImageUsageFlagBits::eSampled;

        vks::Image attachment = context.createImage(imageCreateInfo);

        vk::ImageViewCreateInfo imageView;
        imageView.viewType = vk::ImageViewType::e2D;
        imageView.format = format;
        imageView.subresourceRange.aspectMask = aspectMask;
        imageView.subresourceRange.levelCount = 1;
        imageView.subresourceRange.layerCount = 1;
        imageView.image = attachment.image;

        attachment.view = device.createImageView(imageView);

        return attachment;
    }

    // Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
    void prepareoffscreenfer() {
        {
            offscreen.extent = size;
            // Color attachments
            // Two floating point color buffers
            offscreen.color[0] =
                createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eInputAttachment);
            offscreen.color[1] =
                createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eInputAttachment);
            // Depth attachment
            offscreen.depth = createAttachment(depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment);

            // Set up separate renderpass with references to the colorand depth attachments
            std::array<vk::AttachmentDescription, 3> attachmentDescs;

            // Formats
            attachmentDescs[0].format = offscreen.color[0].format;
            attachmentDescs[0].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            attachmentDescs[1].format = offscreen.color[1].format;
            attachmentDescs[1].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            attachmentDescs[2].format = offscreen.depth.format;
            attachmentDescs[2].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

            // Init attachment properties
            for (uint32_t i = 0; i < 3; ++i) {
                attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
                attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
                attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            }

            std::vector<vk::AttachmentReference> colorReferences{
                { 0, vk::ImageLayout::eColorAttachmentOptimal },
                { 1, vk::ImageLayout::eColorAttachmentOptimal },
            };

            vk::AttachmentReference depthReference{ 2, vk::ImageLayout::eDepthStencilAttachmentOptimal };

            vk::SubpassDescription subpass;
            subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpass.pColorAttachments = colorReferences.data();
            subpass.colorAttachmentCount = 2;
            subpass.pDepthStencilAttachment = &depthReference;

            // Use subpass dependencies for attachment layput transitions
            std::array<vk::SubpassDependency, 2> dependencies{
                vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                       vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
                                       vk::DependencyFlagBits::eByRegion },
                vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                       vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead, vk::AccessFlagBits::eMemoryRead,
                                       vk::DependencyFlagBits::eByRegion },
            };

            vk::RenderPassCreateInfo renderPassInfo;
            renderPassInfo.attachmentCount = (uint32_t)attachmentDescs.size();
            renderPassInfo.pAttachments = attachmentDescs.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = (uint32_t)dependencies.size();
            renderPassInfo.pDependencies = dependencies.data();

            offscreen.renderPass = device.createRenderPass(renderPassInfo);

            std::array<vk::ImageView, 3> attachments{
                offscreen.color[0].view,
                offscreen.color[1].view,
                offscreen.depth.view,
            };

            offscreen.frameBuffer = device.createFramebuffer(vk::FramebufferCreateInfo{ {},
                                                                                        offscreen.renderPass,
                                                                                        (uint32_t)attachments.size(),
                                                                                        attachments.data(),
                                                                                        offscreen.extent.width,
                                                                                        offscreen.extent.height,
                                                                                        1 });

            // Create sampler to sample from the color attachments
            vk::SamplerCreateInfo sampler;
            sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
            sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
            sampler.maxAnisotropy = 1.0f;
            sampler.maxLod = 1.0f;
            sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
            offscreen.sampler = device.createSampler(sampler);
        }

        // Bloom separable filter pass
        {
            filterPass.extent = size;

            // Color attachments

            // Two floating point color buffers
            filterPass.color[0] = createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment);

            // Set up separate renderpass with references to the colorand depth attachments
            std::array<vk::AttachmentDescription, 1> attachmentDescs;

            // Init attachment properties
            attachmentDescs[0].loadOp = vk::AttachmentLoadOp::eClear;
            attachmentDescs[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attachmentDescs[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attachmentDescs[0].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            attachmentDescs[0].format = filterPass.color[0].format;

            std::vector<vk::AttachmentReference> colorReferences{ { 0, vk::ImageLayout::eGeneral } };

            vk::SubpassDescription subpass;
            subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpass.colorAttachmentCount = (uint32_t)colorReferences.size();
            subpass.pColorAttachments = colorReferences.data();

            // Use subpass dependencies for attachment layput transitions
            std::array<vk::SubpassDependency, 2> dependencies{
                vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                       vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
                                       vk::DependencyFlagBits::eByRegion },
                vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                       vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead, vk::AccessFlagBits::eMemoryRead,
                                       vk::DependencyFlagBits::eByRegion },
            };

            filterPass.renderPass = device.createRenderPass(vk::RenderPassCreateInfo{ {},
                                                                                      (uint32_t)attachmentDescs.size(),
                                                                                      attachmentDescs.data(),
                                                                                      1,
                                                                                      &subpass,
                                                                                      (uint32_t)dependencies.size(),
                                                                                      dependencies.data() });

            std::array<vk::ImageView, 1> attachments{ filterPass.color[0].view };

            filterPass.frameBuffer = device.createFramebuffer(
                { {}, filterPass.renderPass, (uint32_t)attachments.size(), attachments.data(), offscreen.extent.width, offscreen.extent.height, 1 });

            // Create sampler to sample from the color attachments
            vk::SamplerCreateInfo sampler;
            sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
            sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
            sampler.maxAnisotropy = 1.0f;
            sampler.maxLod = 1.0f;
            sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
            filterPass.sampler = device.createSampler(sampler);
        }
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildDeferredCommandBuffer() {
        if (!offscreen.cmdBuffer) {
            offscreen.cmdBuffer = context.createCommandBuffer();
        }

        // Create a semaphore used to synchronize offscreen rendering and usage
        if (!offscreen.semaphore) {
            offscreen.semaphore = device.createSemaphore({});
        }

        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        // Clear values for all attachments written in the fragment sahder
        std::array<vk::ClearValue, 3> clearValues{
            vks::util::clearColor(glm::vec4(0)),
            vks::util::clearColor(glm::vec4(0)),
            vk::ClearDepthStencilValue{ 1.0, 0 },
        };

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = offscreen.renderPass;
        renderPassBeginInfo.framebuffer = offscreen.frameBuffer;
        renderPassBeginInfo.renderArea.extent = offscreen.extent;
        renderPassBeginInfo.clearValueCount = (uint32_t)clearValues.size();
        renderPassBeginInfo.pClearValues = clearValues.data();

        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.extent));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.extent));

        vk::DeviceSize offsets[1] = { 0 };

        // Skybox
        if (displaySkybox) {
            offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.models, 0, descriptorSets.skybox, nullptr);
            offscreen.cmdBuffer.bindVertexBuffers(0, models.skybox.vertices.buffer, { 0 });
            offscreen.cmdBuffer.bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
            offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
            offscreen.cmdBuffer.drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
        }

        // 3D object
        const auto& model = models.objects[models.objectIndex];
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.models, 0, descriptorSets.object, nullptr);
        offscreen.cmdBuffer.bindVertexBuffers(0, model.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(model.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.reflect);
        offscreen.cmdBuffer.drawIndexed(model.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();
        offscreen.cmdBuffer.end();
    }

    void loadAssets() override {
        // Models
        models.skybox.loadFromFile(context, getAssetPath() + "models/cube.obj", models.vertexLayout, 0.05f);
        std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
        objectNames = { "Sphere", "Teapot", "Torusknot", "Venus" };
        models.objects.resize(filenames.size());
        for (size_t i = 0; i < filenames.size(); ++i) {
            auto& model = models.objects[i];
            const auto& file = filenames[i];
            model.loadFromFile(context, getAssetPath() + "models/" + file, models.vertexLayout, 0.05f * (file == "venus.fbx" ? 3.0f : 1.0f));
        }

        // Load HDR cube map
        textures.envmap.loadFromFile(context, getAssetPath() + "textures/hdr/uffizi_cube.ktx", vk::Format::eR16G16B16A16Sfloat);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 4 },
            { vk::DescriptorType::eCombinedImageSampler, 6 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 4, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.models = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.models = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.models });

        // Bloom filter
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.bloomFilter = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.bloomFilter = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.bloomFilter });

        // G-Buffer composition
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.composition = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.composition = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.composition });
    }

    void setupDescriptorSets() {
        // 3D object descriptor set
        descriptorSets.object = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.models })[0];
        // Sky box descriptor set
        descriptorSets.skybox = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.models })[0];
        // Bloom filter
        descriptorSets.bloomFilter = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.bloomFilter })[0];
        // Composition descriptor set
        descriptorSets.composition = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.composition })[0];

        std::vector<vk::DescriptorImageInfo> colorDescriptors = {
            { offscreen.sampler, offscreen.color[0].view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { offscreen.sampler, offscreen.color[1].view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { offscreen.sampler, filterPass.color[0].view, vk::ImageLayout::eShaderReadOnlyOptimal },
        };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            vk::WriteDescriptorSet{ descriptorSets.object, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.matrices.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.object, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.envmap.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.object, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.skybox, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.matrices.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.skybox, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.envmap.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.skybox, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.bloomFilter, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[0] },
            vk::WriteDescriptorSet{ descriptorSets.bloomFilter, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[1] },
            vk::WriteDescriptorSet{ descriptorSets.composition, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[0] },
            vk::WriteDescriptorSet{ descriptorSets.composition, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[2] },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Final fullscreen composition pass pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.composition, renderPass };
        pipelineBuilder.depthStencilState = { false };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        // Empty vertex input state, full screen triangles are generated by the vertex shader
        pipelineBuilder.loadShader(getAssetPath() + "shaders/hdr/composition.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/hdr/composition.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.composition = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Bloom pass
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;

        pipelineBuilder.loadShader(getAssetPath() + "shaders/hdr/bloom.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/hdr/bloom.frag.spv", vk::ShaderStageFlagBits::eFragment);

        // Set constant parameters via specialization constants
        uint32_t dir = 1;
        vk::SpecializationMapEntry specializationMapEntry{ 0, 0, sizeof(uint32_t) };
        vk::SpecializationInfo specializationInfo{ 1, &specializationMapEntry, sizeof(dir), &dir };
        pipelineBuilder.shaderStages[1].pSpecializationInfo = &specializationInfo;
        pipelines.bloom[0] = pipelineBuilder.create(context.pipelineCache);
        // Second blur pass (into separate framebuffer)
        pipelineBuilder.renderPass = filterPass.renderPass;
        dir = 0;
        pipelines.bloom[1] = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Object rendering pipelines
        // Binding description
        pipelineBuilder.vertexInputState.appendVertexLayout(models.vertexLayout);
        pipelineBuilder.renderPass = renderPass;

        // Skybox pipeline (background cube)
        blendAttachmentState.blendEnable = VK_FALSE;
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
        pipelineBuilder.layout = pipelineLayouts.models;
        pipelineBuilder.renderPass = offscreen.renderPass;
        pipelineBuilder.colorBlendState.blendAttachmentStates.resize(2);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/hdr/gbuffer.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/hdr/gbuffer.frag.spv", vk::ShaderStageFlagBits::eFragment);

        // Set constant parameters via specialization constants
        uint32_t shadertype = 0;
        specializationInfo = { 1, &specializationMapEntry, sizeof(shadertype), &shadertype };
        pipelineBuilder.shaderStages[0].pSpecializationInfo = &specializationInfo;
        pipelineBuilder.shaderStages[1].pSpecializationInfo = &specializationInfo;
        pipelines.skybox = pipelineBuilder.create(context.pipelineCache);
        // Object rendering pipeline
        shadertype = 1;
        // Enable depth test and write
        pipelineBuilder.depthStencilState = { true };
        // Flip cull mode
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelines.reflect = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Matrices vertex shader uniform buffer
        uniformBuffers.matrices = context.createUniformBuffer(uboVS);
        // Params
        uniformBuffers.params = context.createUniformBuffer(uboParams);

        updateUniformBuffers();
        updateParams();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.modelview = camera.matrices.view;
        memcpy(uniformBuffers.matrices.mapped, &uboVS, sizeof(uboVS));
    }

    void updateParams() { memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams)); }

    void draw() override {
        prepareFrame();
        context.submit(offscreen.cmdBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreen.semaphore);
        renderWaitSemaphores = { offscreen.semaphore };
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareUniformBuffers();
        prepareoffscreenfer();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        buildDeferredCommandBuffer();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.comboBox("Object type", &models.objectIndex, objectNames)) {
                updateUniformBuffers();
                buildDeferredCommandBuffer();
            }
            if (ui.inputFloat("Exposure", &uboParams.exposure, 0.025f)) {
                updateParams();
            }
            if (ui.checkBox("Bloom", &bloom)) {
                buildCommandBuffers();
            }
            if (ui.checkBox("Skybox", &displaySkybox)) {
                buildDeferredCommandBuffer();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
