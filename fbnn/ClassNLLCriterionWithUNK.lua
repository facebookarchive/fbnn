-- Copyright 2004-present Facebook. All Rights Reserved.
-- Author: Michael Mathieu <myrhev@fb.com>

-- This file implements a wrapper for ClassNLLCriterion, which ignores a
-- specific label <unk>

require 'nn'

local ClassNLLCriterionWithUNK, parent =
    torch.class('nn.ClassNLLCriterionWithUNK', 'nn.Criterion')

function ClassNLLCriterionWithUNK:__init(unk_index, sizeAverage)
    self.unk_index = unk_index
    self.crit = nn.ClassNLLCriterion()
    if sizeAverage ~= nil then
        self.crit.sizeAverage = sizeAverage
    end
    self.tmp = torch.LongTensor()
    self.output = 0 --TODO: this doesn't work with CudaTensors
    self.gradInputExtra = torch.Tensor()
    self.gradInput = self.crit.gradInput
    self.tensortype = torch.Tensor():type()
end

function ClassNLLCriterionWithUNK:cuda()
    nn.Criterion.cuda(self)
    self.crit:cuda()
    self.tensortype = 'torch.CudaTensor'
    return self
end

function ClassNLLCriterionWithUNK:updateOutput(input, target)
    local n = 0
    if input:dim() == 1 then
        if ((type(target) == 'number') and (target ~= self.unk_index)) or
           ((type(target) ~= 'number') and (taget[1] ~= self.unk_index))
        then
           self.output = self.crit:updateOutput(input, target)
           n = 1
        end
    else -- minibatch
        assert(input:dim() == 2)
        assert(target:dim() == 1)
        self.tmp:resize(target:size())
        self.use_unk = false
        if self.tensortype == 'torch.CudaTensor' then
            self.use_unk = (self.unk_index ~= nil)
        elseif self.unk_index then
            torch.eq(self.tmp, target, self.unk_index)
            if self.tmp:sum() > 0 then
                self.use_unk = true
            end
        end
        if self.use_unk then -- to go faster, do only that if needed
            self.output = 0
            for i = 1,input:size(1) do
                if target[i] ~= self.unk_index then
                    self.output = self.output +
                        self.crit:updateOutput(input[i], target[i])
                    n = n + 1
                end
            end
        else
            self.output = self.crit:updateOutput(input, target)
            n = target:size(1)
        end
    end
    return self.output, n
end

function ClassNLLCriterionWithUNK:updateGradInput(input, target)
    if input:dim() == 1 then
        if ((type(target) == 'number') and (target ~= self.unk_index)) or
           ((type(target) ~= 'number') and (taget[1] ~= self.unk_index))
        then
            self.gradInput = self.crit:updateGradInput(input, target)
        end
    else --minibatch
        assert(input:dim() == 2)
        assert(target:dim() == 1)
        if self.use_unk then
            self.gradInputExtra:resizeAs(input)
            for i = 1,input:size(1) do
                if target[i] ~= self.unk_index then
                    self.gradInputExtra[i]
                    :copy(self.crit:updateGradInput(input[i], target[i]))
                else
                    self.gradInputExtra[i]:zero()
                end
            end
            self.gradInput = self.gradInputExtra
        else
            self.gradInput = self.crit:updateGradInput(input, target)
        end
    end
    return self.gradInput
end
