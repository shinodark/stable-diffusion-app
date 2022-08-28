import os
import fcntl
from PIL import Image

# Image preprocessing
def preprocess(image, sizemax):
    w, h = sizemax
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    image.thumbnail(sizemax, resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

# Lock helpers
LOCK_PATH = "/var/lock/applock"

def acquire_lock(lock_path):
  lock_file_fd = None
  fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
  try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except (IOError, OSError):
    pass
  else:
    lock_file_fd = fd
  if lock_file_fd is None:
    os.close(fd)
  return lock_file_fd

def release_lock(lock_file_fd):
    fcntl.flock(lock_file_fd, fcntl.LOCK_UN)
    os.close(lock_file_fd)
    return None  