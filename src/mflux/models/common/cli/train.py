from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.training.runner import TrainingRunner
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.utils.exceptions import StopTrainingException


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Finetune a LoRA adapter")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    config_path = args.config
    resume_path = args.resume
    base_lora_path = args.base_lora
    if config_path is not None and not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    if resume_path is not None and not resume_path.exists():
        parser.error(f"Checkpoint not found: {resume_path}")
    if base_lora_path is not None and not base_lora_path.exists():
        parser.error(f"Base LoRA file not found: {base_lora_path}")
    if base_lora_path is not None and resume_path is not None:
        parser.error("--base-lora cannot be used with --resume")
    if base_lora_path is not None and config_path is None:
        parser.error("--base-lora requires --config")

    if args.dry_run:
        TrainingSpec.resolve(
            config_path=str(config_path) if config_path is not None else None,
            resume_path=str(resume_path) if resume_path is not None else None,
            create_output_dir=False,
        )
        print("✅ Training config validated.")
        return

    try:
        TrainingRunner.train(
            config_path=str(config_path) if config_path is not None else None,
            resume_path=str(resume_path) if resume_path is not None else None,
            base_lora_path=str(base_lora_path) if base_lora_path is not None else None,
        )
    except StopTrainingException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
