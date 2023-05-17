from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
prompt = "a photo of a race car on a bridge over train tracks"
image = pipe(prompt).images[0]

image.save("Car-train-bridge.png")
