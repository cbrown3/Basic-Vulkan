#pragma once

#include <GLFW/glfw3.h>
#include <vector>

struct Attachment
{
	VkAttachmentDescription description;
	VkAttachmentReference reference;
};

class RenderPass
{
public:
	RenderPass();
	~RenderPass();

	/*Setting Render Descriptions*/
	//Sets the color attachment for each image view
	void SetColorAttachment(VkFormat format, VkSampleCountFlagBits samples);
	//Sets the depth attachment for each image view
	void SetDepthAttachment(VkFormat format, VkSampleCountFlagBits samples);
	//Sets the color attachment resolve for each image view
	void SetColorAttachmentResolve(VkFormat format, VkSampleCountFlagBits samples);
	//Set subpass for each image view
	void SetSubpass();
	//Set the dependency for each image view
	void SetDependency();
	//Set the render pass info for each image view
	void SetRenderInfo();

	//Calls all the previous fucntions and creates the render pass
	void init(VkRenderPass renderPass, VkDevice device, VkFormat format, VkSampleCountFlagBits samples);

private:
	Attachment colorAttachment;
	Attachment depthAttachment;
	Attachment colorAttachmentResolve;
	VkSubpassDescription subpass;
	VkSubpassDependency dependency;
	VkRenderPassCreateInfo renderInfo;
};

