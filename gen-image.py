from diffuser import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusions-v-1-5")

prompt = "At night in Singapore, the silouette of six athletes in running attire are visible jogging around Marina Bay"

image = pipe(prompt).images[0]
image.save("weekly_run_images.png")

