#pragma once

#include "context.hpp"

namespace vks { namespace renderpass {

template <typename T>
static void updatePointerFromVector(const std::vector<T>& vector, uint32_t& count, const T*& data) {
    if (vector.empty()) {
        count = 0;
        data = nullptr;
    } else {
        count = static_cast<uint32_t>(vector.size());
        data = vector.data();
    }
}

template <typename T>
static void updatePointerFromVector(const std::vector<T>& vector, const T*& data) {
    if (vector.empty()) {
        data = nullptr;
    } else {
        data = vector.data();
    }
}

struct SubpassDescription : public vk::SubpassDescription {
    std::vector<vk::AttachmentReference> inputAttachments;
    std::vector<vk::AttachmentReference> colorAttachments;
    std::vector<vk::AttachmentReference> resolveAttachments;
    std::vector<uint32_t> preserveAttachments;

    void update() {
        if (!resolveAttachments.empty() && resolveAttachments.size() != colorAttachments.size()) {
            throw std::runtime_error("Resolve attachments vector must be empty or equal in size to the color attachments vector");
        }
        updatePointerFromVector(inputAttachments, inputAttachmentCount, pInputAttachments);
        // Set the resolve attachments first, since the color attachments will overwrite the count
        updatePointerFromVector(resolveAttachments, pResolveAttachments);
        updatePointerFromVector(colorAttachments, colorAttachmentCount, pColorAttachments);
        updatePointerFromVector(preserveAttachments, preserveAttachmentCount, pPreserveAttachments);
    }

private:
};

struct RenderPassCreateInfo : public vk::RenderPassCreateInfo {
    using Parent = vk::RenderPassCreateInfo;
    RenderPassCreateInfo() {}

    std::vector<SubpassDescription> subpasses;
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::SubpassDependency> dependencies;

    void update() {
        updatePointerFromVector(attachments, attachmentCount, pAttachments);
        updatePointerFromVector(dependencies, dependencyCount, pDependencies);
    }
};

}}  // namespace vks::renderpass
