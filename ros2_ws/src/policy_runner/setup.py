import glob

from setuptools import find_packages, setup

package_name = "policy_runner"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/topics.yaml"]),
        (f"share/{package_name}/static", glob.glob("static/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="asu",
    maintainer_email="im.zhangxc@gmail.com",
    description=(
        "Subscribes camera images + joint_states over ROS2 and serves a "
        "browser page showing which streams are live -- a data-availability "
        "check before training/inference."
    ),
    license="TODO",
    entry_points={
        "console_scripts": [
            "web_monitor = policy_runner.web_monitor:main",
            "act_infer_mujoco = policy_runner.act_infer_mujoco:main",
            "ood_build_reference = policy_runner.ood_reference_builder:main",
        ],
    },
)
