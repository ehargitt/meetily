# Diarization test fixtures

## `ami_es2004a_795s_60s.*`

A 60 s excerpt (795.0 s to 855.0 s) of AMI Meeting Corpus meeting **ES2004a**, headset mix, with
all four participants speaking (FEE013, FEE016, MEE014, MEO015; 15 to 20 s of speech each).

| File | Content |
|---|---|
| `ami_es2004a_795s_60s.wav` | 16 kHz mono 16-bit PCM, samples copied unchanged from `ES2004a.Mix-Headset.wav` |
| `ami_es2004a_795s_60s.rttm` | Reference speaker turns clipped to the excerpt and shifted so the excerpt starts at 0 s |
| `ami_es2004a_795s_60s.expected_turns.json` | Turns from the diarization prototype (`diar-proto run --step 48000`, 3 s step) on this excerpt |

The reference turns come from the `only_words` RTTM of
[pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup)
(`only_words/rttms/test/ES2004a.rttm`), which is derived from the AMI word-level annotations.

### Attribution and license

The AMI Meeting Corpus and its annotations are released under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
as stated on the corpus license page: <https://groups.inf.ed.ac.uk/ami/corpus/license.shtml>.

- Source: AMI Meeting Corpus, meeting ES2004a (<https://groups.inf.ed.ac.uk/ami/corpus/>),
  AMI Consortium.
- Modifications: cut to 60 s; RTTM clipped and re-timed to the excerpt. The audio samples are
  otherwise unchanged.
- No warranties are given, as set out in the license.
