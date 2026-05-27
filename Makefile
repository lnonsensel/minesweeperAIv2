.PHONY: train eval play test

train:
	python main.py

play:
	python play_minesweeper.py

test:
	pytest test.py -v

debug:
	MINESWEEPER_DEBUG=1 python main.py
