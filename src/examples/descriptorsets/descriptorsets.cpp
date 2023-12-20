/*
* Vulkan Example - Using descriptor sets for passing data to shader stages
*
* Relevant code parts are marked with [POI]
*
* Copyright (C) 2018 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase {
public:
    bool animate{ true };

    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
    } };

    struct Cube {
        struct Matrices {
            glm::mat4 projection;
            glm::mat4 view;
            glm::mat4 model;
        } matrices;
        vk::DescriptorSet descriptorSet;
        vks::texture::Texture2D texture;
        vks::Buffer uniformBuffer;
        glm::vec3 rotation;

        void destroy() {
            texture.destroy();
            uniformBuffer.destroy();
        }
    };
    std::array<Cube, 2> cubes;

    struct Models {
        vks::model::Model cube;
    } models;

    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;

    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        title = "Using descriptor Sets";
        settings.overlay = true;
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, size, 0.1f, 512.0f);
        camera.setRotation({ 0.0f, 0.0f, 0.0f });
        camera.setTranslation({ 0.0f, 0.0f, -5.0f });
    }

    ~VulkanExample() {
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        models.cube.destroy();
        for (auto cube : cubes) {
            cube.uniformBuffer.destroy();
            cube.texture.destroy();
        }
    }

    void getEnabledFeatures() override {
        if (context.deviceFeatures.samplerAnisotropy) {
            context.enabledFeatures.samplerAnisotropy = VK_TRUE;
        };
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        drawCmdBuffer.bindVertexBuffers(0, models.cube.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(models.cube.indices.buffer, 0, vk::IndexType::eUint32);

        /*
        [POI] Render cubes with separate descriptor sets
        */
        for (const auto& cube : cubes) {
            // Bind the cube's descriptor set. This tells the command buffer to use the uniform buffer and image set for this cube
            drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, cube.descriptorSet, nullptr);
            drawCmdBuffer.drawIndexed(models.cube.indexCount, 1, 0, 0, 0);
        }
    }

    void loadAssets() override {
        models.cube.loadFromFile(context, getAssetPath() + "models/cube.dae", vertexLayout, 1.0f);
        cubes[0].texture.loadFromFile(context, getAssetPath() + "textures/crate01_color_height_rgba.ktx");
        cubes[1].texture.loadFromFile(context, getAssetPath() + "textures/crate02_color_height_rgba.ktx");
    }

    /*
        [POI] Set up descriptor sets and set layout
    */
    void setupDescriptors() {
        /*

            Descriptor set layout
            
            The layout describes the shader bindings and types used for a certain descriptor layout and as such must match the shader bindings

            Shader bindings used in this example:

            VS:
                layout (set = 0, binding = 0) uniform UBOMatrices ...

            FS :
                layout (set = 0, binding = 1) uniform sampler2D ...;

        */

        std::array<vk::DescriptorSetLayoutBinding, 2> setLayoutBindings{};

        /*
        Binding 0: Uniform buffers (used to pass matrices matrices)
        */
        setLayoutBindings[0] =
            vk::DescriptorSetLayoutBinding{ // Shader binding point
                                            0,
                                            // This is a uniform buffer
                                            vk::DescriptorType::eUniformBuffer,
                                            // Binding contains one element (can be used for array bindings)
                                            1,
                                            // Accessible from the vertex shader only (flags can be combined to make it accessible to multiple shader stages)
                                            vk::ShaderStageFlagBits::eVertex
            };

        /*
        Binding 1: Combined image sampler (used to pass per object texture information)
        */
        setLayoutBindings[1] = vk::DescriptorSetLayoutBinding{ // Shader binding point
                                                               1,
                                                               // This is a image buffer
                                                               vk::DescriptorType::eCombinedImageSampler,
                                                               // Binding contains one element (can be used for array bindings)
                                                               1,
                                                               // Accessible from the fragment shader only
                                                               vk::ShaderStageFlagBits::eFragment
        };

        // Create the descriptor set layout
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });

        /*

            Descriptor pool

            Actual descriptors are allocated from a descriptor pool telling the driver what types and how many
            descriptors this application will use

            An application can have multiple pools (e.g. for multiple threads) with any number of descriptor types
            as long as device limits are not surpassed

            It's good practice to allocate pools with actually required descriptor types and counts

        */

        std::array<vk::DescriptorPoolSize, 2> descriptorPoolSizes;

        // Uniform buffers : 1 for scene and 1 per object (scene and local matrices)
        descriptorPoolSizes[0] = vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 + static_cast<uint32_t>(cubes.size()) };

        // Combined image samples : 1 per mesh texture
        descriptorPoolSizes[1] = vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(cubes.size()) };

        // Create the global descriptor pool
        // Max. number of descriptor sets that can be allocted from this pool (one per object)
        descriptorPool = device.createDescriptorPool(
            { {}, static_cast<uint32_t>(descriptorPoolSizes.size()), static_cast<uint32_t>(descriptorPoolSizes.size()), descriptorPoolSizes.data() });

        /*

            Descriptor sets

            Using the shared descriptor set layout and the descriptor pool we will now allocate the descriptor sets.

            Descriptor sets contain the actual descriptor fo the objects (buffers, images) used at render time.

        */

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

        for (auto& cube : cubes) {
            // Allocates an empty descriptor set without actual descriptors from the pool using the set layout
            cube.descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

            // Update the descriptor set with the actual descriptors matching shader bindings set in the layout
            /*
            Binding 0: Object matrices Uniform buffer
            */
            writeDescriptorSets.push_back({ cube.descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cube.uniformBuffer.descriptor });
            /*
            Binding 1: Object texture
            */
            // Images use a different descriptor strucutre, so we use pImageInfo instead of pBufferInfo
            writeDescriptorSets.push_back({ cube.descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cube.texture.descriptor });
        }
        // Execute the writes to update descriptors for ALL sets
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        /*
        [POI] Create a pipeline layout used for our graphics pipeline
        */
        // The pipeline layout is based on the descriptor set layout we created above
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });

        auto builder = vks::pipelines::GraphicsPipelineBuilder(device, pipelineLayout, renderPass);
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // Vertex bindings and attributes
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.loadShader(getAssetPath() + "shaders/descriptorsets/cube.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/descriptorsets/cube.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipeline = builder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Vertex shader matrix uniform buffer block
        for (auto& cube : cubes) {
            cube.uniformBuffer = context.createUniformBuffer<glm::mat4>({});
        }
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        cubes[0].matrices.model = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f));
        cubes[1].matrices.model = glm::translate(glm::mat4(1.0f), glm::vec3(1.5f, 0.5f, 0.0f));

        for (auto& cube : cubes) {
            cube.matrices.projection = camera.matrices.perspective;
            cube.matrices.view = camera.matrices.view;
            cube.matrices.model = glm::rotate(cube.matrices.model, glm::radians(cube.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
            cube.matrices.model = glm::rotate(cube.matrices.model, glm::radians(cube.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
            cube.matrices.model = glm::rotate(cube.matrices.model, glm::radians(cube.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
            memcpy(cube.uniformBuffer.mapped, &cube.matrices, sizeof(cube.matrices));
        }
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareUniformBuffers();
        setupDescriptors();
        preparePipelines();
        buildCommandBuffers();
        prepared = true;
    }

    void update(float deltaTime) override {
        if (animate) {
            cubes[0].rotation.x += 2.5f * frameTimer;
            if (cubes[0].rotation.x > 360.0f)
                cubes[0].rotation.x -= 360.0f;
            cubes[1].rotation.y += 2.0f * frameTimer;
            if (cubes[1].rotation.x > 360.0f)
                cubes[1].rotation.x -= 360.0f;
            viewUpdated = true;
        }

        ExampleBase::update(deltaTime);
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            ui.checkBox("Animate", &animate);
        }
    }
};

VULKAN_EXAMPLE_MAIN()
