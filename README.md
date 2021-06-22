## What
It includes the most basic [Parakeet](https://github.com/iclementine/Parakeet) English Text To Speach docker image

## Why
Parakeet generates very good quality of English Speach. However, you need certain knowledge to download and to configure it. Here is a repackage of it as a docker image, which can be used easily.


## Usage
```
echo "Text needs to be Convert" > __input__.txt
docker run --rm -v $(pwd):/workspace privapps/tts-parakeet

# output is at __out__.wav and __out__.mp3
```

### Note
It is recommended to have 3G+ RAM to use it.