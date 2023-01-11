import filters
import torch_filters

# from filters import test_imp

# from Ensrf import Ensrf


def main():
    filters.Senkf()
    torch_filters.EnsrfTorch()

    print("import successful")


if __name__ == "__main__":
    main()
