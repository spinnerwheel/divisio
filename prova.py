from PIL import Image

p = {}

new_image = Image.new('RGB', (100, 100), color = 'red')
p['a'] = new_image
p['b'] = new_image
p['c'] = new_image

print(p['a'])