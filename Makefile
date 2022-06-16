all:
	@echo "Hi :). Nothing is implemented in here yet."

install:
	pip install flake8 black[jupyter] isort parameterized
	pip install -e .

pretty:
	isort .
	black --line-length 100 .

test:
	flake8

uninstall:
	pip install pip-autoremove
	pip-autoremove trainlib -y
	pip uninstall pip-autoremove -y

clean:
	rm -r .ipynb_checkpoints
	rm -r data/.ipynb_checkpoints
	rm -r tests/.ipynb_checkpoints
	rm -r trainlib/.ipynb_checkpoints
