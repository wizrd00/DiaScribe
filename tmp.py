#!/usr/bin/python3
import argparse
import sys, os
from collections.abc import Iterator
from typing import BinaryIO

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError


def manage_args() :
	parser = argparse.ArgumentParser(prog = "agent", add_help = True)
	parser.add_argument("--file", type = str, required = True, help = "The audio file")
	parser.add_argument("--api-key", type = str, required = True, help = "The OpenAI API Key")
	parser.add_argument("--base-url", type = str, required = True, help = "The OpenAI base url")
	parser.add_argument("--output", type = str, required = False, help = "Specify output file, if you don't specify this argument, output will be stdout by default", default = sys.stdout.fileno())
	return parser.parse_args()

args = manage_args()


class Diarization :
	def __init__(self, client : OpenAI , file : str) :
		self.client = client
		self.file = file
		self.result = None

	@property
	def client(self) -> OpenAI :
		return self._client

	@client.setter
	def client(self, value : OpenAI) -> None :
		if (value.is_closed) :
			raise Exception("ERROR : the OpenAI client closed")
		else :
			self._client = value

	@property
	def file(self) -> BinaryIO :
		return self._file

	@file.setter
	def file(self, value : str) -> None :
		if (not os.path.exists(value)) :
			raise Exception(f"ERROR : file in path {value} doesn't exists")
		else :
			self._file = open(value, "rb")

	def diarize(self) -> list :
		self.result = self.client.audio.transcriptions.create(
			model = "gpt-4o-transcribe-diarize",
			file = self.file,
			response_format = "diarized_json",
			chunking_strategy = "auto"
		)
		return self.result.segments

	def __del__(self) :
		if (not self.file.closed) :
			self.file.close()


class Transcription(Diarization) :
	def __init__(self, client : OpenAI, file : str) :
		super(client, file)

	def transcribe(self) -> list :
		self.result = self.client.audio.transcriptions.create(
			model = "whisper-1",
			file = self.file,
			response_format = "verbose_json"
		)
		return self.result.segments

def merge(diaz_segments : list, trans_segments : list) -> list[tuple[float, float, str, str]] :
	merged = []
	for text in trans_segments :
		best_speaker = None
		best_overlap = 0
		for speak in diaz_segments :
			overlap = min(text.end, speak.end) - max(text.start, speak.start)
			if (overlap > best_overlap) :
				best_overlap = overlap
				best_speaker = speak.speaker
		merged.append((text.start, text.end, best_speaker, text.text))

def clean(output, client) :
	print("cleaning context ... ", end = "", flush = True)
	output.close()
	client.close()
	print("DONE")

def main() :
	if (not os.path.exists(args.file)) :
		raise ValueError(f"ERROR : the file \"{args.file}\" doesn't exist")
	try :
		client = OpenAI(
			api_key = args.api_key,
			base_url = args.base_url
		)
		diaz_obj = Diarization(client, args.file)
		diaz = diaz_obj.diarize()
		trans_obj = Transcription(client, args.file)
		trans = trans_obj.transcribe()
	except APIConnectionError as err :
		print("Connection Error : ", err)
	except APITimeoutError as err :
		print("Timeout Error : ", err)
	except RateLimitError as err :
		print("Rate Limit Error : ", err)
	except APIStatusError as err :
		print("Server Error : ", err)
	except Exception as err :
		print("OpenAI Error : ", err)
	else :
		for start, end, speaker, text in merge(diaz, trans) :
			print(f"[{start:.2f}->{end:.2f}] {speaker} : {text}")
	finally :
		raise SystemExit()
	output = open(args.output, "w")
	output.close()

try :
	main()
except KeyboardInterrupt :
	print(f"\x1b[31mKeyboard Interrupt \x1b[0m")
	raise SystemExit()
