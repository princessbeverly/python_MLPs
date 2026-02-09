YELLOW:str = '\x1b[93m\x1b[1m'
RESET:str = '\x1b[0m'

IMAGE_WIDTH:int = 28
PIXELS_PER_IMAGE:int = 784

class Image:
	'''
	28x28 Grayscale Image

	Attributes:
		- label:int
		- pixels:list[float]

	Each label is a non-zero integer in [0, 9];
	Each pixel's brightness is represented by a real number in [0.0, 1.0].
	'''

	def __init__( self, label:int, pixels:list[float] ) -> None:
		'''
		Creates a new image.
		'''

		self.label:int = label
		self.pixels:list[float] = pixels

	def display( self ):
		'''
		Displays a projection of the image in the terminal.

		The label is displayed on the top left with the image itself encased in a frame.
		'''

		# The projection string is built up piece by piece,
		# and finally displayed in the terminal.
		projection:str =  f'\n╔╡ {YELLOW}{self.label}{RESET} ╞═{'══' * (IMAGE_WIDTH - 3)}╗'
		for i in range(PIXELS_PER_IMAGE):
			# Left border
			if i % IMAGE_WIDTH == 0:
				projection += f'\n║{YELLOW}'

			# Pixel values map to certain block characters of varying brightnesses.
			if self.pixels[i] == 0:
				projection += '  '
			elif self.pixels[i] <= 0.25:
				projection += '░░'
			elif self.pixels[i] <= 0.50:
				projection += '▒▒'
			elif self.pixels[i] <= 0.75:
				projection += '▓▓'
			else:
				projection += '██'

			# Right border
			if i % IMAGE_WIDTH == IMAGE_WIDTH - 1:
				projection += f'{RESET}║'
		projection += f'\n╚{'══' * IMAGE_WIDTH}╝'

		print(projection)