#define GLFW_INCLUDE_VULKAN
#include <array>
#include "RenderPass.h"



RenderPass::RenderPass()
{
	//Initializae all descriptions and references
	colorAttachment.description, colorAttachment.reference,
		depthAttachment.description, depthAttachment.reference,
		colorAttachmentResolve.description, colorAttachmentResolve.reference,
		subpass, dependency, renderInfo = {};
}

RenderPass::~RenderPass()
{
}

void RenderPass::SetColorAttachment(VkFormat format, VkSampleCountFlagBits samples)
{
	colorAttachment.description.format = format;
	colorAttachment.description.samples = samples;
	colorAttachment.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.description.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	colorAttachment.reference.attachment = 0;
	colorAttachment.reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
}

void RenderPass::SetDepthAttachment(VkFormat format, VkSampleCountFlagBits samples)
{
	depthAttachment.description.format = format;
	depthAttachment.description.samples = samples;
	depthAttachment.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.description.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	depthAttachment.reference.attachment = 1;
	depthAttachment.reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
}

void RenderPass::SetColorAttachmentResolve(VkFormat format, VkSampleCountFlagBits samples)
{
	colorAttachmentResolve.description.format = format;
	colorAttachmentResolve.description.samples = samples;
	colorAttachmentResolve.description.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentResolve.description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentResolve.description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentResolve.description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	colorAttachmentResolve.reference.attachment = 2;
	colorAttachmentResolve.reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
}

void RenderPass::SetSubpass()
{
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachment.reference;
	subpass.pDepthStencilAttachment = &depthAttachment.reference;
	subpass.pResolveAttachments = &colorAttachmentResolve.reference;
}

void RenderPass::SetDependency()
{
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
}

void RenderPass::SetRenderInfo()
{
	std::array<VkAttachmentDescription, 3> attachments = { colorAttachment.description, depthAttachment.description, colorAttachmentResolve.description };
	renderInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderInfo.pAttachments = attachments.data();
	renderInfo.subpassCount = 1;
	renderInfo.pSubpasses = &subpass;
	renderInfo.dependencyCount = 1;
	renderInfo.pDependencies = &dependency;
}

void RenderPass::init(VkRenderPass renderPass, VkDevice device, VkFormat format, VkSampleCountFlagBits samples)
{
	SetColorAttachment(format, samples);
	SetDepthAttachment(format, samples);
	SetColorAttachmentResolve(format, samples);
	SetSubpass();
	SetDependency();
	SetRenderInfo();

	if (vkCreateRenderPass(device, &renderInfo, nullptr, &renderPass) != VK_SUCCESS)
		throw std::runtime_error("failed to create render pass!");
}
