import docker, signal, time

from docker.client import DockerClient
from docker.models.containers import Container

TIMEOUT_DURATION = 10
START_UP_DELAY = 3


class timeout:
    def __init__(self, seconds=TIMEOUT_DURATION, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def get_container(ctr_name: str, image_name: str, **kwargs) -> Container:
    """
    Reset docker container with given name, or create new container with given name if it does not exist

    Returns:
        Container: reference to docker container object
    """
    client = docker.from_env()
    all_containers = [container.name for container in client.containers.list(all=True)]
    container = None
    if ctr_name in all_containers:
        # Reset container via cleanup script if it exists
        container = client.containers.get(ctr_name)
        if container.status != "running":
            container.start()
    else:
        # Only keep relevant kwargs
        for key in kwargs.copy().keys():
            if key not in ["command", "environment", "ports", "volumes"]:
                del kwargs[key]
                
        # Create + return new container from custom image
        image = client.images.get(image_name)
        container = client.containers.run(
            image=image,
            name=ctr_name,
            detach=True,
            tty=True,
            **kwargs)
    time.sleep(START_UP_DELAY)
    
    # Check if container was created successfully
    if not container:
        raise RuntimeError(f"Failed to create and start `{ctr_name}` container successfully")
    
    return container