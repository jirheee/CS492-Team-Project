export enum ProcessType {
  Train,
  Battle
}

export interface RandomBattleRequest {
  agentUuid: string;
  opponent: 'Random' | string;
}
