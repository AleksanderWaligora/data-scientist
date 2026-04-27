#!pip install uuid

#!pip install accelerate diffusers controlnet_aux

import torch
from controlnet_aux import CannyDetector
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image, make_image_grid
import uuid

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
    varient="fp16")


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Yntec/AbsoluteReality",
    controlnet=controlnet,
    torch_dtype=torch.float16)

pipe.load_ip_adapter("h94/IP-Adapter",
                     subfolder="models",
                     weight_name="ip-adapter_sd15.bin")

pipe.enable_model_cpu_offload()



class ImageGenerator:

  def __init__(self,path_img_adapter,path_img_canny,width,height,is_canny=True):
      self.__ip_adapt_img = load_image(path_img_adapter)
      self.__height=height
      self.__width=width
      self.__generated_images=None
      if is_canny:
        img = load_image(path_img_canny).resize((width,height))
        canny = CannyDetector()
        self.__canny_img = canny(img,detect_resolution=512,image_resolution=768)
      else:
        self.__canny_img = load_image(path_img_canny).resize((width,height))

  def show_canny_image(self):
      return self.__canny_img

  def show_adapt_image(self):
      return self.__ip_adapt_img

  def generate_images(self,prompt,negative_prompt):
    images = pipe(prompt = prompt,
              negative_prompt = negative_prompt,
              height = self.__height,
              width = self.__width,
              ip_adapter_image=self.__ip_adapt_img,
              image=self.__canny_img,
              guidance_scale=6,
              controlnet_conditioning_scale=0.7,
              num_interface_steps=20,
              num_images_per_prompt=3).images


    self.__generated_images = images

  def show_generated_images(self):
    if self.__generated_images is None:
      raise Exception("You need to generate images first")
    return make_image_grid(self.__generated_images,cols=3,rows=1)

  def save_generated_images(self,save_path):
    if self.__generated_images is None:
        raise Exception("You need to generate images first")
    else:
      for image in self.__generated_images:
          image_id = str(uuid.uuid4())
          image.save(f"{save_path}/{image_id}.jpg")

