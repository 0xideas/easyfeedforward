import argparse
import inspect

def opt_meta_function(function):

	params = inspect.getargspec(function)
	variables = params.args

	ap = argparse.ArgumentParser()

	shorts_in_use = []
	short_to_long = {}
	for v in variables:
		for letter in v:
			if letter not in shorts_in_use:
				short = "-" + letter
				shorts_in_use += [short[1:]]
				short_to_long[v] = short[1:]
				break

		long_ = "-" + v
		ap.add_argument(short, long_, required=True)

	args = vars(ap.parse_args())
	long_args = {}
	for key in short_to_long.keys():
		long_args[key] = args[short_to_long[key]]
	
	function(**long_args)

def test_opt_meta_function(word1, word2):
	uselessvariable = None
	print(word1)
	print(word2)
	print("This works!")

if __name__ == '__main__':
	opt_meta_function(test_opt_meta_function)