import yaml

with open('configs/citylearn_maddpg.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('╔' + '═' * 76 + '╗')
print('║' + ' AJUSTES APLICADOS PARA EVITAR KeyboardInterrupt '.center(76) + '║')
print('╚' + '═' * 76 + '╝')
print()
print('CAMBIOS EN configs/citylearn_maddpg.yaml:')
print('─' * 78)
print('  batch_size:       512 → 256        (reducido 50%)')
print('  update_every:     10 → 20          (updates menos frecuentes)')
print('  updates_per_step: 2 → 1            (menos iteraciones de gradient)')
print()
print('CAMBIOS EN src/maddpg_tesis/maddpg/maddpg.py:')
print('─' * 78)
print('  ✓ Try-except en backward() del critic')
print('  ✓ Try-except en backward() del actor')
print('  ✓ Recuperación automática de KeyboardInterrupt')
print()
print('═' * 78)
print('CONFIGURACIÓN ACTUALIZADA:')
print('─' * 78)
maddpg_cfg = config['maddpg']
print(f'  batch_size:       {maddpg_cfg["batch_size"]}')
print(f'  update_every:     {maddpg_cfg["update_every"]}')
print(f'  updates_per_step: {maddpg_cfg["updates_per_step"]}')
print('═' * 78)
