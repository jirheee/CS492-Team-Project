import {
  BaseEntity,
  Column,
  CreateDateColumn,
  Entity,
  PrimaryGeneratedColumn
} from 'typeorm';
import { TrainStatus } from '../types/nn';

@Entity()
export default class Agent extends BaseEntity {
  @PrimaryGeneratedColumn('uuid')
  public uuid: string;

  @CreateDateColumn()
  public createdAt: Date;

  @Column()
  public name: string;

  @Column({ default: TrainStatus.NOT_TRAINED })
  public trainStatus: TrainStatus;
}
