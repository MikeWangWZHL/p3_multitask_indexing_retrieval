import promptsource.templates
from collections import defaultdict
import re

# def get_original_promptsource_template_name()


if __name__ == "__main__":
    input_path = '/cephfs/user/xiaomanpan/data/tmp/bigscience_P3 - overall.tsv'

    with open(input_path, 'r') as f:
        first_line = f.readline()
        assert first_line.startswith('category')
        for line in f:
            tmp = line.rstrip('\n').split('\t')
            task = tmp[1]
            dataset = tmp[2]

            if '/' in task:
                dataset_name, dataset_config_name = task.split('/')
                prompts = promptsource.templates.DatasetTemplates(
                    dataset_name, dataset_config_name)

                if dataset_config_name == 'ARC-Challenge':
                    dataset_config_name = 'ARC_Challenge'
                if dataset_config_name == 'ARC-Easy':
                    dataset_config_name = 'ARC_Easy'

                template_name = dataset.replace(f'{dataset_config_name}_', '').replace(f'{dataset_name}_', '').replace('_score_eval', '').strip('_')
            else:
                dataset_name = task
                prompts = promptsource.templates.DatasetTemplates(dataset_name)
                template_name = dataset.replace(f'{dataset_name}_', '').replace('_score_eval', '').strip('_')
                if task == 'anli':
                    template_name = template_name.replace('_r1', '').replace('_r2', '').replace('_r3', '')

            res = {}
            for i in prompts.templates:
                tn = prompts.templates[i].name.replace('-', '_').replace(' ', '_').replace('/', '_').replace('___', '_')
                tn = re.sub(r"[^\w\d'\s\_]+", '', tn).strip('_')
                res[tn] = prompts.templates[i].metadata.original_task

            try:
                is_original_task = res[template_name]
            except KeyError:
                print(res)
                print(template_name)
                print(tmp)
                exit()

            if is_original_task:
                print(f'{dataset}\t1')
            else:
                print(f'{dataset}\t')
