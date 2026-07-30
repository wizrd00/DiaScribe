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
		if (value.is_closed()) :
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
		super().__init__(client, file)

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
	return merged

def main() :
	if (not os.path.exists(args.file)) :
		print(f"ERROR : the file \"{args.file}\" doesn't exist")
	if (os.path.getsize(args.file) > 26214400) :
		print("ERROR : the maximum size of file must be 25MiB")
		raise SystemExit()
	try :
		print("Creating OpenAI client object...")
		client = OpenAI(
			api_key = args.api_key,
			base_url = args.base_url
		)
		print("Creating Diarization object...")
		diaz_obj = Diarization(client, args.file)
		print("diarizing...")
		diaz = diaz_obj.diarize()
		print("Creating Transcription object...")
		trans_obj = Transcription(client, args.file)
		print("transcribing...")
		trans = trans_obj.transcribe()
		print("done, start merging results")
	except APIConnectionError as err :
		print("\nConnection Error : ", err)
	except APITimeoutError as err :
		print("\nTimeout Error : ", err)
	except RateLimitError as err :
		print("\nRate Limit Error : ", err)
	except APIStatusError as err :
		print("\nServer Error : ", err)
	except Exception as err :
		print("\nOpenAI Error : ", err)
	else :
		output = open(args.output, "w")
		print()
		for start, end, speaker, text in merge(diaz, trans) :
			print(f"[{start:.2f}->{end:.2f}] Speaker {speaker} : {text}", file = output)
		client.close()
		output.close()
	finally :
		raise SystemExit()

main()
