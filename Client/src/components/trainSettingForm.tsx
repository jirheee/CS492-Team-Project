import { Grid, Button, GridItem, Flex } from '@chakra-ui/react';
import NumberInputWithFieldname from './Inputs/NumberInputWithFieldName';

const TrainSettingForm = ({ onSubmit }) => {
  return (
    <Flex w="full" flexDir="column" m={5}>
      <Grid
        as="form"
        // marginLeft="auto"
        // m={5}
        minW="200px"
        h="full"
        templateColumns="1fr 1fr"
        onSubmit={onSubmit}
      >
        <GridItem colSpan={1}>
          <NumberInputWithFieldname
            label="Learning Rate"
            id="lr"
            min={0.00001}
            max={0.1}
            defaultValue={0.0001}
            step={0.00001}
          />
        </GridItem>
        <GridItem colSpan={1}>
          <NumberInputWithFieldname
            label="Buffer Size"
            id="buffer_size"
            min={5000}
            max={10000}
            defaultValue={5000}
            step={500}
          />
        </GridItem>
        <GridItem colSpan={1}>
          <NumberInputWithFieldname
            label="Batch Size"
            id="batch_size"
            min={64}
            max={256}
            defaultValue={64}
            step={32}
          />
        </GridItem>
        <GridItem colSpan={1}>
          <NumberInputWithFieldname
            label="Epochs"
            id="epochs"
            min={30}
            max={300}
            defaultValue={150}
            step={30}
          />
        </GridItem>
        <GridItem colSpan={2}>
          <Button mt={4} colorScheme="teal" type="submit" w="full" h="full">
            Start Training
          </Button>
        </GridItem>
      </Grid>
    </Flex>
  );
};

export default TrainSettingForm;
