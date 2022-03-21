# AI Speaker using raspberry pi

- Raspberry pi zero W
- [Raspberry Pi Zero W camera](https://www.devicemart.co.kr/goods/view?no=1376528)
- [ReSpeaker 2 Mics Pi HAT](https://www.devicemart.co.kr/goods/view?no=1383296)


## Setup the device
- Step 1. Install the latest Raspberry Pi OS on your Pi

- Step 2. Setup about ReSpeaker 2 Mics
    - Install driver [Link](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT_Raspberry/#driver-installation-and-configuration)
    - Install the necessary dependencies  
    ``` bash
    sudo apt-get install wiringpi
    sudo apt-get install portaudio19-dev libatlas-base-dev
    pip install spidev rpi.gpio pyaudio
    ```

- Step 3. Test ReSpeaker 2
    ``` bash
    # LED on chip works
    python test/pixels.py 

    # Button works
    python test/button.py 
    
    # Check input device id of seeed-2mic-voicecard and output Device id of playback
    python test/get_device_index.py 

    # Change RESPEAKER_INDEX to input device id
    vim test/record.py 
    
    # Change RESPEAKER_INDEX to output Device id
    vim test/play.py 

    # Record 5 seconds --> Save output.wav
    python test/record.py 

    # Play output.wav
    python test/play.py output.wav
    ```

- Step 4. Setup the Pi Camera [Link](https://picamera.readthedocs.io/en/release-1.13/quickstart.html#pi-zero)


- Step 5. Install OpenCV 
    ``` bash
    sudo apt install libcblas-dev
    sudo apt install python3-opencv
    ```