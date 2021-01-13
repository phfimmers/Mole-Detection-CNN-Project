<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/selmaesen/Mole-Detection-CNN-Project/">
    <img src="assets/immo_logo.svg" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Mole-Detection-CNN-Project Online</h3>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Website</a></li>
    <li><a href="#api">API</a></li>
	<li><a href="#preprocess">Preprocess</a></li>
	<li><a href="#model">Model</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


The purpose of the project is to develo a tool that would be able to detect moles that need to be handle by doctors.
We use and train the VGG model to detect when the mole is dangerous.
the project will be available on a simple web page where the user could upload a picture of the mole and see the result.
The project will be upload on internet with flask, doker and heroku. 



### Built With

* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [Docker](https://www.docker.com/)
* [Heroku](https://www.heroku.com/)



<!-- GETTING STARTED -->
## Getting Started

To work with this API, you have two options. Either work directly with the API at [this URL](https://?????.herokuapp.com/), either build it yourself from the sources and deploy it in a Docker container on Heroku as it is explained in the next subsection.

### Prerequisites

You'll need the packages/software described above.

### Installation

#### HEROKU

* **Install the Heroku CLI:**
  * The Heroku Command Line Interface (CLI) makes it easy to create and manage your Heroku apps directly from the terminal.
It’s an essential part of using Heroku.
  ```sh
  sudo snap install --classic heroku
  ```
* **Deployment on Heroku:**
  * Heroku favours Heroku CLI therefore using command line is (ensure the CLI is up-to-date) crucial at this step. 
  ```sh
  heroku login
  ```
  * After logging in to the respective Heroku account, the container needs to be registered with Heroku using 
  ```sh
  heroku container:login
  ```
  * Once the container has been registered, a Heroku repo would be required to push the container which could be created : 
  ```sh
  heroku create <yourapplicationname>
  ```
  **NOTE**: If there is no name stated after '_create_', a random name will be assigned.
  
  * When there is an application repo to push the container, it is time to push the container to web : 
  ```sh
  heroku container:push web --app <yourapplicationname>
  ```
  * Following the 'container:push' , the container should be released on web to be visible with 
  ```sh
  heroku container:release web --app <yourapplicationname>
  ```
  * If the container has been released properly, it is available to see using 
  ```sh
  heroku open --app <yourapplicationname>
  ```
  * Logging is also critical especially if the application is experiencing errors : 
  ```sh
  heroku logs --tail <yourapplicationname>
  ```


**IMPORTANT NOTE:** While with _localhost_ and _Docker_ it is not mandatory to specify the PORT, if one would like to deploy on Heroku, the port needs to be specified within the 'app.py' to avoid crashes.

## API

Our REST API is deployed on Heroku, using a Docker container. It is available at [this address](https://predict-keras-api.herokuapp.com/).

Now, let's describe our simple little API's routes and endpoints and the different HTTP methods that can be used.

### `/`

This route is used with a `GET` method and returns a string "alive" in case the server is running and alive.

### `/predict`

There are two endpoints for this route. The most important one is reached with a `POST` method but it is also accessible with a `GET` method. Let's further discuss these methods.

#### `GET`

This endpoint does not need any input. It returns a string explaining the input data and their format that the `POST` method expects.


#### `POST`

This endpoint is the main one of this API. With it, you will be able to query a price prediction giving abritrary real estate property features. It needs and returns specifically formatted inputs and outputs that will be described below.

##### **Input**

T
##### **Output**



<!-- Authors -->
## Authors
* [**Selma Esen**](https://github.com/selmaesen/) - *BeCoder* 
* [**Philippe Fimmers**](https://github.com/phfimmers/) - *BeCoder* 
* [**Christophe Giets**](https://github.com/gietsc/) - *BeCoder* 
* [**Mikael Dominguez**](https://github.com/wiiki09) - *BeCoder and Dancer*

Project Link: [https://github.com/selmaesen/Mole-Detection-CNN-Project](https://github.com/selmaesen/Mole-Detection-CNN-Project)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
