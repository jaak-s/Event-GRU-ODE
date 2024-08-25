import setuptools

exec(open("event_gruode/version.py").read())

setuptools.setup(
    name="event_gruode",
    version=__version__,
    author="Jaak Simm",
    author_email="jaak.simm@gmail.com",
    description="Event modeling with GRU-ODE",
    long_description="Time continuous neural network based models to estimate probability of events.",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas", "sklearn", "tqdm", "torch>=1.4.0", "tensorboard"],
)

