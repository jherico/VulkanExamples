/*
* Vulkan Example - Push descriptors
*
* Note: Requires a device that supports the VK_KHR_push_descriptor extension
*
* Push descriptors apply the push constants concept to descriptor sets. So instead of creating 
* per-model descriptor sets (along with a pool for each descriptor type) for rendering multiple objects, 
* this example uses push descriptors to pass descriptor sets for per-model textures and matrices 
* at command buffer creation time.
*
* Copyright (C) 2018 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase
{
public:
    bool animate = true;

    vk::DispatchLoaderDynamic dispatcher;
    vk::PhysicalDevicePushDescriptorPropertiesKHR pushDescriptorProps{};

    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
    });

    struct Cube {
        vks::texture::Texture2D texture;
        vks::Buffer uniformBuffer;
        glm::vec3 rotation;
        glm::mat4 modelMat;
    };
    std::array<Cube, 2> cubes;

    struct Models {
        vks::model::Model cube;
    } models;

    struct UniformBuffers {
        vks::Buffer scene;
    } uniformBuffers;

    struct UboScene {
        glm::mat4 projection;
        glm::mat4 view;
    } uboScene;

    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample()  {
        title = "Push descriptors";
        settings.overlay = true;
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, size, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -5.0f));

        // Disable validation layers.  
        // Depending on the driver the vkGetPhysicalDeviceProperties2KHR may be present, but not vkGetPhysicalDeviceProperties2
        // However, if the validation layers are turned on, the dispatcher finds the vkGetPhysicalDeviceProperties2 function anyway
        // (presumably provided by the validation layers regardless of whether the underlying implementation actually has the 
        // function) and promptly crashes 
        // Bug filed against the validation layers here:  https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/issues/2562
        context.setValidationEnabled(false);
        // Enable extension required for push descriptors
        context.requireExtensions({ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
        context.requireDeviceExtensions({ VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME });
    }

    ~VulkanExample()
    {
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        models.cube.destroy();
        for (auto cube : cubes) {
            cube.uniformBuffer.destroy();
            cube.texture.destroy();
        }
        uniformBuffers.scene.destroy();
    }

    void getEnabledFeatures() override 
    {
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

        // Render two cubes using different descriptor sets using push descriptors
        for (const auto& cube : cubes) {

            // Instead of preparing the descriptor sets up-front, using push descriptors we can set (push) them inside of a command buffer
            // This allows a more dynamic approach without the need to create descriptor sets for each model
            // Note: dstSet for each descriptor set write is left at zero as this is ignored when ushing push descriptors
            std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
            writeDescriptorSets.reserve(3);

            // Scene matrices
            writeDescriptorSets.push_back({ nullptr, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.scene.descriptor });
            // Model matrices
            writeDescriptorSets.push_back({ nullptr, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cube.uniformBuffer.descriptor });
            // Texture
            writeDescriptorSets.push_back({ nullptr, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cube.texture.descriptor });

            drawCmdBuffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, writeDescriptorSets, dispatcher);
            drawCmdBuffer.drawIndexed(models.cube.indexCount, 1, 0, 0, 0);
        }
    }


    void loadAssets() override 
    {
        models.cube.loadFromFile(context, getAssetPath() + "models/cube.dae", vertexLayout);
        cubes[0].texture.loadFromFile(context, getAssetPath() + "textures/crate01_color_height_rgba.ktx");
        cubes[1].texture.loadFromFile(context, getAssetPath() + "textures/crate02_color_height_rgba.ktx");
    }

    void setupDescriptorSetLayout()
    {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        // Setting this flag tells the descriptor set layouts that no actual descriptor sets are allocated but instead pushed at command buffer creation time
        descriptorSetLayout = device.createDescriptorSetLayout({ vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });

        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void preparePipelines()
    {
        // The pipeline layout is based on the descriptor set layout we created above
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });

        auto builder = vks::pipelines::GraphicsPipelineBuilder(device, pipelineLayout, renderPass);
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // Vertex bindings and attributes
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.loadShader(getAssetPath() + "shaders/pushdescriptors/cube.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/pushdescriptors/cube.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipeline = builder.create(context.pipelineCache);
    }

    void prepareUniformBuffers()
    {

        // Vertex shader scene uniform buffer block
        uniformBuffers.scene = context.createUniformBuffer<UboScene>({});

        // Vertex shader cube model uniform buffer blocks
        for (auto& cube : cubes) {
            cube.uniformBuffer = context.createUniformBuffer<glm::mat4>({});
        }

        updateUniformBuffers();
        updateCubeUniformBuffers();
    }

    void updateUniformBuffers()
    {
        uboScene.projection = camera.matrices.perspective;
        uboScene.view = camera.matrices.view;
        memcpy(uniformBuffers.scene.mapped, &uboScene, sizeof(UboScene));
    }

    void updateCubeUniformBuffers()
    {
        cubes[0].modelMat = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f));
        cubes[1].modelMat = glm::translate(glm::mat4(1.0f), glm::vec3( 1.5f, 0.5f, 0.0f));

        for (auto& cube : cubes) {
            cube.modelMat = glm::rotate(cube.modelMat, glm::radians(cube.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
            cube.modelMat = glm::rotate(cube.modelMat, glm::radians(cube.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
            cube.modelMat = glm::rotate(cube.modelMat, glm::radians(cube.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
            memcpy(cube.uniformBuffer.mapped, &cube.modelMat, sizeof(glm::mat4));
        }

        if (animate) {
            cubes[0].rotation.x += 2.5f * frameTimer;
            if (cubes[0].rotation.x > 360.0f)
                cubes[0].rotation.x -= 360.0f;
            cubes[1].rotation.y += 2.0f * frameTimer;
            if (cubes[1].rotation.x > 360.0f)
                cubes[1].rotation.x -= 360.0f;
        }
    }

    void prepare() override
    {
        ExampleBase::prepare();
        /*
            Extension specific functions
        */

        // The push descriptor update function is part of an extension so it has to be manually loaded 
        // The DispatchLoaderDynamic class exposes all known extensions (to the current SDK version)
        // and handles dynamic loading.  It must be initialized with an instance in order to fetch 
        // instance-level extensions and with an instance and device to expose device level extensions.  
        dispatcher.init(context.instance, &vkGetInstanceProcAddr, context.device, &vkGetDeviceProcAddr);

        // Get device push descriptor properties (to display them)
        if (context.deviceProperties.apiVersion >= VK_MAKE_VERSION(1, 1, 0)) {
            pushDescriptorProps = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDevicePushDescriptorPropertiesKHR>(dispatcher).get<vk::PhysicalDevicePushDescriptorPropertiesKHR>();
        } else {
            pushDescriptorProps = context.physicalDevice.getProperties2KHR<vk::PhysicalDeviceProperties2KHR, vk::PhysicalDevicePushDescriptorPropertiesKHR>(dispatcher).get<vk::PhysicalDevicePushDescriptorPropertiesKHR>();
        }

        /*
            End of extension specific functions
        */

        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        buildCommandBuffers();
        prepared = true;
    }


    void update(float deltaTime) override {
        ExampleBase::update(deltaTime);
        if (animate) {
            cubes[0].rotation.x += 2.5f * frameTimer;
            if (cubes[0].rotation.x > 360.0f)
                cubes[0].rotation.x -= 360.0f;
            cubes[1].rotation.y += 2.0f * frameTimer;
            if (cubes[1].rotation.x > 360.0f)
                cubes[1].rotation.x -= 360.0f;
            updateCubeUniformBuffers();
        }
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override 
    {
        if (ui.header("Settings")) {
            ui.checkBox("Animate", &animate);
        }
        if (ui.header("Device properties")) {
            ui.text("maxPushDescriptors: %d", pushDescriptorProps.maxPushDescriptors);
        }
    }
};

VULKAN_EXAMPLE_MAIN()