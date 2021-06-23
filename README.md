## What
It includes the most basic [Parakeet](https://github.com/iclementine/Parakeet) English Text To Speach, including [docker image](https://github.com/privapps/TTS-Parakeet/tree/docker) and [Github Action](https://github.com/privapps/TTS-Parakeet/tree/main)

## Why
Parakeet generates very good quality of English Speech. However, you need certain knowledge to download and to configure it. Here is a repackage of it, which can be used easily.


## Docker Usage
```
echo "Text needs to be Convert" > __input__.txt
docker run --rm -v $(pwd):/workspace privapps/tts-parakeet

# output is at __out__.wav and __out__.mp3
```

## Github Action Usage
See [action.yml](https://github.com/privapps/TTS-Parakeet/blob/main/action.yml)
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