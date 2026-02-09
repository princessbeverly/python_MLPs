# Source for understanding ubyte file content:
# https://medium.com/theconsole/do-you-really-know-how-mnist-is-stored-600d69455937

import Image

CYAN: str = '\x1b[96m'
GREEN: str = '\x1b[92m\x1b[1m'
RESET: str = '\x1b[0m'

IMAGE_WIDTH:int = 28
PIXELS_PER_IMAGE:int = 784

IDX1_UBYTE_INDEX:int = 8
IDX3_UBYTE_INDEX:int = 16

TRAINING_IMAGE_COUNT:int = 60000
TESTING_IMAGE_COUNT:int = 10000

def get_byte_values( file_name:str, start_index:int, end_index:int ) -> list[int]:
	'''
	Reads a file and retrieves a specified range of byte content in the form of integers.
	'''

	# The file is open; the contents are read; and the file is closed.
	print(f'{CYAN}\tReading bytes from [{file_name}]{RESET}.. ', end = '')
	file = open(file_name, 'rb')
	raw_values:list[int] = [int(raw_byte) for raw_byte in file.read()]
	file.close()
	print(f'{GREEN}DONE.{RESET}')

	# IMPORTANT: Check if we're asking for more bytes than exist
	total_bytes = len(raw_values)
	if end_index > total_bytes:
		print(f'\n{CYAN}WARNING: Requested {end_index} bytes but file only has {total_bytes} bytes{RESET}')
		print(f'{CYAN}Adjusting to read all available bytes from index {start_index}{RESET}')
		end_index = total_bytes

	# Relevant bytes are in [start_index, end_index),
	# so only those are returned.
	return raw_values[start_index: end_index]

def get_labels( file_name:str, label_count:int ) -> list[int]:
	'''
	Retrieves the labels in a label-containing idx1-ubyte file.
	'''

	# Bytes 0-7 are unimportant.
	# Bytes 8-END encode label data: one byte per label.

	# The labels are returned in a vector of integers.
	return get_byte_values(file_name, IDX1_UBYTE_INDEX, IDX1_UBYTE_INDEX + label_count)

def get_pixel_values( file_name:str, image_count:int ) -> list[list[float]]:
	'''
	Retrieves the pixel values for every image in an image-containing idx3-ubyte file.
	'''

	# Bytes 0-15 are unimportant.
	# Bytes 16-END encode pixel data: one byte per pixel.
	# These bytes need to be grouped into 784-pixel arrays.

	# Each set of pixels is stored in a vector of vectors,
	# with each vector (set of pixels) containing 784 pixel brightness values.
	# The brightness values are scaled down from [0, 255] to [0.0, 1.00].

	# Calculate how many bytes we need
	bytes_needed = image_count * PIXELS_PER_IMAGE
	print(f'{CYAN}\tExpecting {bytes_needed} pixel bytes for {image_count} images{RESET}')

	pixel_bytes:list[int] = get_byte_values(file_name, IDX3_UBYTE_INDEX, IDX3_UBYTE_INDEX + bytes_needed)

	# Check if we got enough data
	actual_bytes = len(pixel_bytes)
	print(f'{CYAN}\tReceived {actual_bytes} pixel bytes{RESET}')

	if actual_bytes < bytes_needed:
		print(f'{CYAN}\tWARNING: Only got {actual_bytes}/{bytes_needed} bytes. Adjusting image count.{RESET}')
		image_count = actual_bytes // PIXELS_PER_IMAGE
		print(f'{CYAN}\tWill process {image_count} complete images{RESET}')

	# Create the pixel values list properly
	pixel_values:list[list[float]] = []

	for i in range(image_count):
		image_pixels:list[float] = []
		start_idx = i * PIXELS_PER_IMAGE
		end_idx = (i + 1) * PIXELS_PER_IMAGE

		# Make sure we don't go out of bounds
		if end_idx > len(pixel_bytes):
			print(f'{CYAN}\tWARNING: Stopping at image {i} due to insufficient data{RESET}')
			break

		for k in range(start_idx, end_idx):
			image_pixels.append(float(pixel_bytes[k]) / 255.0)

		pixel_values.append(image_pixels)

	return pixel_values

class MNIST_Dataset:
	'''
	Complete Original MNIST Dataset

	Attributes:
		- training_images:list[Image]
		- testing_images:list[Image]
	'''

	def __init__( self, folder_name:str ) -> None:
		'''
		Creates a dataset for MNIST data.

		It is assumed that the file names were unchanged from the original MNIST dataset file set:
			1. t10k-labels.idx1-ubyte
			2. t10k-images.idx3-ubyte
			3. train-labels.idx1-ubyte
			4. train-images.idx3-ubyte
		'''

		# Initialize empty lists
		print(f'{CYAN}\nCreating image blanks.{RESET}')
		self.training_images:list[Image.Image] = []
		self.testing_images:list[Image.Image] = []
		print(f'{GREEN}Blanks created.{RESET}')

		# The labels are gathered next.
		print(f'{CYAN}\nGathering labels.{RESET}')
		training_labels:list[int] = get_labels(f'{folder_name}/train-labels.idx1-ubyte', TRAINING_IMAGE_COUNT)
		testing_labels:list[int] = get_labels(f'{folder_name}/t10k-labels.idx1-ubyte', TESTING_IMAGE_COUNT)
		print(f'{GREEN}Labels gathered.{RESET}')

		# The pixel values are gathered last.
		print(f'{CYAN}\nGathering pixel data.{RESET}')
		training_pixel_values:list[list[float]] = get_pixel_values(f'{folder_name}/train-images.idx3-ubyte', TRAINING_IMAGE_COUNT)
		testing_pixel_values:list[list[float]] = get_pixel_values(f'{folder_name}/t10k-images.idx3-ubyte', TESTING_IMAGE_COUNT)
		print(f'{GREEN}Data gathered.{RESET}')

		# The images are stored as Image objects for ease of access and display.
		print(f'{CYAN}\nSetting labels and data as Image objects.{RESET}')

		# Use the actual number of images we have (in case file was incomplete)
		actual_training_count = min(len(training_labels), len(training_pixel_values))
		actual_testing_count = min(len(testing_labels), len(testing_pixel_values))

		for i in range(actual_training_count):
			self.training_images.append(Image.Image(training_labels[i], training_pixel_values[i]))

		for i in range(actual_testing_count):
			self.testing_images.append(Image.Image(testing_labels[i], testing_pixel_values[i]))

		print(f'{GREEN}Images set.{RESET}')
		print(f'{GREEN}Training images: {len(self.training_images)}{RESET}')
		print(f'{GREEN}Testing images: {len(self.testing_images)}{RESET}')

		print(f'{GREEN}\nThe MNIST dataset is ready.{RESET}')

# TESTING
if __name__ == '__main__':
	dataset: MNIST_Dataset = MNIST_Dataset('mnist_data')

	print('\nSample from training images:')
	for t in dataset.training_images[10:13]:
		t.display()

	print('\nSample from testing images:')
	for t in dataset.testing_images[10:13]:
		t.display()