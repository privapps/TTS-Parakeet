## What
It includes the most basic [Parakeet](https://github.com/iclementine/Parakeet) English Text To Speach, including [docker image]() and [Github Action]()

## Why
Parakeet generates very good quality of English Speech. However, you need certain knowledge to download and to configure it. Here is a repackage of it, which can be used easily.


## Docker Usage
```
echo "Text needs to be Convert" > __input__.txt
docker run --rm -v $(pwd):/workspace privapps/tts-parakeet

# output is at __out__.wav and __out__.mp3
```

## Github Action Usage
```
jobs:
  job:
    steps:
      - uses: privapps/TTS-Parakeet@main
        with:
          text: 'text'
          format: 'mp3'
		  content: '/tmp/output.mp3'
```

### Note
It is recommended to have 3G+ RAM to use it.