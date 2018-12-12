#define GLFW_INCLUDE_VULKAN
#include "SwapChain.h"

SwapChain::SwapChain()
{
}


SwapChain::~SwapChain()
{

}

void SwapChain::update(GLFWwindow* window, VkSurfaceKHR surface, VkPhysicalDevice physicalDevice, VkDevice device)
{
	//load the swap chain setting supported
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(surface, physicalDevice);

	//select the surface format, presentation mode, and resolution(extent)
	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	VkExtent2D extent = chooseSwapExtent(window, swapChainSupport.capabilities);

	//load the number of images in the swap chain, checking if there aren't any images to load
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		imageCount = swapChainSupport.capabilities.maxImageCount;

	//create and set the info for the swap chain
	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;

	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	//use 1 unless you are planning on creating a stereoscopic 3D application
	createInfo.imageArrayLayers = 1;
	//defines how we plan on rendering the images, here we render them directly or additively
	//for example, if it is used for post-processing, you would use a different bit field
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	//sets the queue families
	QueueFamilyIndices indices = findQueueFamilies(surface, physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsFamily != indices.presentFamily) {
		//the image can be used across multiple queue families
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		//sets the index of queue families and queue families being used
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		//the image can be used by one queue family at a time
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;
	}
	//sets the supoprt for different transforms on images
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	//sets how the surface blends with the window
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	//sets presentation mode
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	//checks to see if swap chain is obsolete for the app, (i.e. resizing the app window)
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	//creates swap chain using info above
	if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		throw std::runtime_error("failed to create swap chain!");

	//retrieves the images and loads them in swapChainImages
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
	swapChainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;
}

void SwapChain::clear(VkDevice device)
{
	for (auto imageView : swapChainImageViews) {
		vkDestroyImageView(device, imageView, nullptr);
	}

	vkDestroySwapchainKHR(device, swapChain, nullptr);
}


//finds which queue families are neccessary
//used in createLogicalDevice, createSwapChain, createCommandPool, and isDeviceSuitable->pickPhysicalDevice
QueueFamilyIndices SwapChain::findQueueFamilies(VkSurfaceKHR surface, VkPhysicalDevice device)
{
	QueueFamilyIndices indices;

	//retrieve list of queue families
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device,
		&queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device,
		&queueFamilyCount, queueFamilies.data());

	//loops through the queue families and checks to see if
	//any of them support VK_QUEUE_GRAPHICS_BIT or presenting
	int i = 0;
	for (const auto& queueFamily : queueFamilies)
	{
		if (queueFamily.queueCount > 0 && queueFamily.queueFlags &
			VK_QUEUE_GRAPHICS_BIT)
		{
			indices.graphicsFamily = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (queueFamily.queueCount > 0 && presentSupport)
		{
			indices.presentFamily = i;
		}

		if (indices.isComplete())
			break;

		i++;
	}

	return indices;
}

//populates the swap chain support details
//used in createSwapChain and isDeviceSuitable->pickPhysicalDevice
SwapChainSupportDetails SwapChain::querySwapChainSupport(VkSurfaceKHR surface, VkPhysicalDevice device)
{
	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0)
	{
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

//used in createSwapChain
VkSurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
	if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
		return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

	for (const auto& availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			return availableFormat;
	}

	//return the first format that works
	return availableFormats[0];
}

//used in createSwapChain
VkPresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes)
{
	VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

	//checks to see if wee can use mailbox/triple buffering, or immediate/double buffering
	for (const auto& availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			return availablePresentMode;
		else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			bestMode = availablePresentMode;
	}

	//use FIFO if nothing else is available
	return bestMode;
}

//the swap extent is the resolution of the images, so we will be picking the resolution that
//best matches the window width and height
//used in createSwapChain
VkExtent2D SwapChain::chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		return capabilities.currentExtent;
	else
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent =
		{
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::max(capabilities.minImageExtent.width,
			std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(capabilities.minImageExtent.height,
			std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}