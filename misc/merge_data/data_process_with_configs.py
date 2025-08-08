#!/usr/bin/env python3
"""
Data processing script using configuration files
Supports defining tasks and processing parameters through YAML configuration files
"""

import argparse
import sys
from pathlib import Path

# Import refactored data processing modules
from data_process import (DataProcessor, ProcessConfig, TaskConfig,
                          load_config_from_yaml, logger)


def main():
    """Main function - configuration file driven version"""
    parser = argparse.ArgumentParser(description='Robot data processing tool')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory (overrides settings in configuration file)')
    parser.add_argument('--output-file', '-f', type=str,
                       help='Output filename (overrides settings in configuration file)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show configuration information, without executing processing')

    args = parser.parse_args()

    # Check if configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file does not exist: {config_path}")
        sys.exit(1)

    try:
        # Load configuration
        config_data = load_config_from_yaml(str(config_path))

        # Create processing configuration
        process_config_data = config_data.get('process_config', {})

        # Command line arguments override configuration file
        if args.output_dir:
            process_config_data['target_dir'] = args.output_dir
        if args.output_file:
            process_config_data['output_filename'] = args.output_file

        process_config = ProcessConfig.from_dict(process_config_data)

        # Create data processor
        processor = DataProcessor(process_config)

        # Add tasks
        tasks_config = config_data.get('tasks', {})
        for task_id, task_info in tasks_config.items():

            task_config = TaskConfig.from_dict(task_info)
            processor.add_task(task_id, task_config)

            logger.info(f"Task added: {task_id}")

        if args.dry_run:
            logger.info("Dry run mode - only show configuration information")
            logger.info(f"Output directory: {process_config.target_dir}")
            logger.info(f"Output file: {process_config.output_filename}")
            logger.info(f"Total tasks: {len(tasks_config)}")
            logger.info(f"Total data files: {len(processor.all_episode_path)}")
            return

        # Process data and save results
        statistics = processor.process_episodes()
        processor.save_results(statistics)
        logger.info("Data processing completed!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()