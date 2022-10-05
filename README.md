# TM Assigment 5
#### Repository for theoretical mechanics' assigment 5

## Manual rendering

1. Make sure you have installed:
    - **Python 3.9+** (make sure command `python3 -V` or `python -V`)
    - **PIP** (make sure command `pip`)
    - **Python Venv** for setup virtual environment (check this [useful link](https://docs.python.org/3/library/venv.html))

2. Clone the repo into your folder:
    ```shell
    git clone https://github.com/someilay/TM_Assigment_5.git
    cd ./TM_Assigment_5
    ```

3. Setup virtual environment:
    ```shell
    python3 -m venv venv
    ```

4. Activate environment ([guide](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments)) 
   and install Manim package ([installation](https://docs.manim.community/en/stable/installation.html)).

5. Render scenes by executing:
    ```shell
    manim -pqh task_1.py Task1
    ```

   Render results for Task1 would be located in `media/videos/task_1/1080p60/Task1.mp4`

   Be ready, rendering can take a couple of minutes
