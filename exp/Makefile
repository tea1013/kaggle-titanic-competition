module = module

gen:
	@mkdir ./v${v}
	@mkdir ./v${v}/$(module)
	@touch ./v${v}/$(module)/__init__.py

	@touch ./v${v}/README.md
	@echo "# v${v}" >> ./v${v}/README.md

	@echo "- [#${v}](./v${v}/README.md)" >> README.md

	@touch ./v${v}/Makefile
	@echo ".PHONY: build" >> ./v${v}/Makefile
	@echo "" >> ./v${v}/Makefile
	@echo "build:" >> ./v${v}/Makefile
	@echo "	python ../encode.py ./$(module)" >> ./v${v}/Makefile
	@echo "	cat ./build/build.py | pbcopy" >> ./v${v}/Makefile

clean:
	sudo rm -rf v${v}/log v${v}/models v${v}/oof v${v}/submission

clean/all:
	sudo rm -rf ../input/*
	sudo rm -rf v*/log v*/models v*/oof v*/submission
