#!/usr/bin/python3
import argparse
import sys, os

def manage_args() :
	parser = argparse.ArgumentParser(prog = "agent", add_help = True)
	parser.add_argument("--file", type = str, required = True, help = "The audio file")
	parser.add_argument("--device", type = str, required = False, help = "Device to Models", metavar = ["cuda", "cpu"], choices = ["cuda", "cpu"], default = "cpu")
	parser.add_argument("--beam", type = int, required = False, help = "Specify beam size of whisper Model", choices = range(1, 11), default = 3)
	parser.add_argument("--whisper-model", type = str, required = True, help = "Specify faster-whisper Model")
	parser.add_argument("--pyannote-model", type = str, required = False, help = "Specify PyAnnote Model", default = "pyannote/speaker-diarization-community-1")
	parser.add_argument("--pyannote-token", type = str, required = True, help = "PyAnnote Access Token")
	parser.add_argument("--output", type = str, required = False, help = "Specify output file, if you don't specify this argument, output will be stdout by default")
	return parser.parse_args()

args = manage_args()

import torch
from collections.abc import Iterator
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from faster_whisper import WhisperModel


class Diarization :
	def __init__(self, model : str, token : str, device : str, file : str) :
		self.model = model
		self.token = token
		self.device = device
		self.file = file
		self.pipeline = Pipeline.from_pretrained(self.model, token = self.token)
		self.pipeline.to(torch.device(self.device))
		self.output = self.pipeline(self.file)

	def diarize(self) -> Iterator[dict] :
		for turn, speaker in self.output.speaker_diarization :
			yield {
				"start" : turn.start,
				"end" : turn.end,
				"speaker" : speaker
			}


class Transcription :
	def __init__(self, model : str, device : str, file : str, beam_size : int) :
		self.model = model
		self.device = device
		self.file = file
		self.beam_size = beam_size
		self.trans = WhisperModel(self.model, device = self.device, compute_type = "int8" if (self.device == "cpu") else "float16")
		self.segments, info = self.trans.transcribe(self.file, beam_size = self.beam_size, word_timestamps = True)
		self.lang = info.language
		self.prob = info.language_probability * 100.0
		self.words = []

	def transcribe(self) -> Iterator[dict] :
		for segment in self.segments :
			for word in segment.words :
				self.words.append(word)

def main() :
	if (not os.path.exists(args.file)) :
		raise ValueError(f"the file \"{args.file}\" doesn't exist")
	output = open(args.output, "w") if (args.output) else sys.stdout
	print("Creating Transcription object...")
	trans = Transcription(args.whisper_model, args.device, args.file, args.beam)
	print(f"The {args.whisper_model} is {trans.prob:.2f}% sure language of audio is {trans.lang}")
	print("Transcripting the audio...")
	trans.transcribe()
	print("Creating Diarization object...")
	diaz = Diarization(args.pyannote_model, args.pyannote_token, args.device, args.file)
	print("Diarization the audio...")
	i = 0
	for speak in diaz.diarize() :
		start_time = speak["start"]
		end_time = speak["end"]
		print(f"\n{speak['speaker']} [{speak['start']:.2f} -> {speak['end']:.2f}] :", end = "\n\t", file = output)
		while (i < len(trans.words) and trans.words[i].start < end_time) :
			if (trans.words[i].start >= start_time) :
				print(trans.words[i].word, end = "", file = output)
			i += 1
		print(file = output)
	if (args.output) :
		output.close()

try :
	main()
except KeyboardInterrupt :
	print(f"\x1b[31mKeyboard Interrupt \x1b[0m")
	raise SystemExit()
