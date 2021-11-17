import {
  BaseEntity,
  Column,
  CreateDateColumn,
  Entity,
  PrimaryGeneratedColumn
} from 'typeorm';

@Entity()
export default class Agent extends BaseEntity {
  @PrimaryGeneratedColumn('uuid')
  public uuid: number;

  @CreateDateColumn()
  public createdAt: Date;

  @Column()
  public name: string;
}
