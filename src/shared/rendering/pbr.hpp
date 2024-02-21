#pragma once

#include <rendering/context.hpp>
#include <rendering/texture.hpp>
#include <rendering/model.hpp>

namespace vkx { namespace pbr {
// Generate a BRDF integration map used as a look-up-table (stores roughness / NdotV)
void generateBRDFLUT(vks::texture::Texture2D& target);
// Generate an irradiance cube map from the environment cube map
void generateIrradianceCube(vks::texture::TextureCubeMap& target,
                            const vks::model::Model& skybox,
                            const vks::model::VertexLayout& vertexLayout,
                            const vk::DescriptorImageInfo& skyboxDescriptor);
// Prefilter environment cubemap
// See https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
void generatePrefilteredCube(vks::texture::TextureCubeMap& target,
                             const vks::model::Model& skybox,
                             const vks::model::VertexLayout& vertexLayout,
                             const vk::DescriptorImageInfo& skyboxDescriptor);
}}  // namespace vkx::pbr
